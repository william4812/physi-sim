#include "io/ChipLayout.hpp"

#include <nlohmann/json.hpp>

#include <fstream>
#include <sstream>
#include <stdexcept>

namespace physi_sim::io {

namespace {

[[noreturn]] void fail(const std::string& what) {
    throw std::runtime_error("ChipLayout: " + what);
}

constexpr double UM = 1e-6;          // authoring length unit -> m
constexpr double W_PER_CM2 = 1e4;    // W/cm^2 -> W/m^2
constexpr double C_TO_K = 273.15;

}  // namespace

// ---------------------------------------------------------------------------
// Material
// ---------------------------------------------------------------------------
double Material::get(const std::string& key, double fallback) const {
    const auto it = props.find(key);
    return it == props.end() ? fallback : it->second;
}

double Material::require(const std::string& key) const {
    const auto it = props.find(key);
    if (it == props.end())
        fail("material '" + name + "' is missing required property '" + key + "'");
    return it->second;
}

// ---------------------------------------------------------------------------
// Region
// ---------------------------------------------------------------------------
bool Region::contains(double px, double py, double pz) const {
    // Half-open [lo, hi) so adjacent regions tile without overlap or gaps at the
    // shared plane -- the same convention the solver's cell centres assume.
    auto in = [](const std::optional<std::array<double, 2>>& r, double v) {
        return !r.has_value() || (v >= (*r)[0] && v < (*r)[1]);
    };
    return in(x, px) && in(y, py) && in(z, pz);
}

double Region::volumetricSource() const {
    if (powerSurfaceWm2 == 0.0) return 0.0;
    if (!z.has_value())
        fail("region '" + name + "' has power but no z_um range to derive a thickness");
    const double thickness = (*z)[1] - (*z)[0];
    if (thickness <= 0.0) fail("region '" + name + "' has non-positive thickness");
    return powerSurfaceWm2 / thickness;      // [W/m^2] / [m] = [W/m^3]
}

// ---------------------------------------------------------------------------
// ChipLayout
// ---------------------------------------------------------------------------
std::array<double, 3> ChipLayout::spacing() const {
    return { size[0] / static_cast<double>(cells[0]),
             size[1] / static_cast<double>(cells[1]),
             size[2] / static_cast<double>(cells[2]) };
}

const Region* ChipLayout::regionAt(double x, double y, double z) const {
    const Region* hit = nullptr;
    for (const auto& r : regions)            // LAST match wins -- see header
        if (r.contains(x, y, z)) hit = &r;
    return hit;
}

const Material& ChipLayout::materialAt(double x, double y, double z) const {
    const Region* r = regionAt(x, y, z);
    if (!r) {
        std::ostringstream oss;
        oss << "no region covers point (" << x*1e6 << ", " << y*1e6 << ", " << z*1e6
            << ") um -- the layout has a hole";
        fail(oss.str());
    }
    const auto it = materials.find(r->material);
    if (it == materials.end())
        fail("region '" + r->name + "' references unknown material '" + r->material + "'");
    return it->second;
}

double ChipLayout::kAt(double x, double y, double z) const {
    return materialAt(x, y, z).k();
}

double ChipLayout::sourceAt(double x, double y, double z) const {
    const Region* r = regionAt(x, y, z);
    return r ? r->volumetricSource() : 0.0;
}

// ---------------------------------------------------------------------------
// Loader
// ---------------------------------------------------------------------------
ChipLayout loadChipLayout(const std::string& path) {
    std::ifstream file(path);
    if (!file.is_open()) fail("cannot open '" + path + "'");

    nlohmann::json j;
    try {
        j = nlohmann::json::parse(file);
    } catch (const std::exception& e) {
        fail("malformed JSON in '" + path + "': " + e.what());
    }

    ChipLayout L;

    // ---- domain -----------------------------------------------------------
    const auto& d = j.at("domain");
    const auto sz = d.at("size_um").get<std::vector<double>>();
    const auto nc = d.at("cells").get<std::vector<std::size_t>>();
    if (sz.size() != 3 || nc.size() != 3) fail("domain.size_um and domain.cells need 3 entries");
    for (int a = 0; a < 3; ++a) {
        if (sz[static_cast<std::size_t>(a)] <= 0.0) fail("domain.size_um must be positive");
        if (nc[static_cast<std::size_t>(a)] < 2)    fail("domain.cells must be >= 2 per axis");
        L.size[static_cast<std::size_t>(a)]  = sz[static_cast<std::size_t>(a)] * UM;
        L.cells[static_cast<std::size_t>(a)] = nc[static_cast<std::size_t>(a)];
    }

    // ---- materials: every key is copied verbatim into the property bag -----
    // This loop is why a new property needs no code change. Whatever numeric
    // fields the JSON declares become available through get()/require().
    for (const auto& [name, body] : j.at("materials").items()) {
        Material m;
        m.name = name;
        for (const auto& [key, val] : body.items())
            if (val.is_number()) m.props[key] = val.get<double>();
        if (m.props.find("k_w_mk") == m.props.end())
            fail("material '" + name + "' must declare k_w_mk");
        L.materials.emplace(name, std::move(m));
    }
    if (L.materials.empty()) fail("no materials declared");

    // ---- regions ----------------------------------------------------------
    auto range = [&](const nlohmann::json& body, const char* key)
        -> std::optional<std::array<double, 2>> {
        if (!body.contains(key)) return std::nullopt;
        const auto v = body.at(key).get<std::vector<double>>();
        if (v.size() != 2 || v[1] <= v[0]) fail(std::string("bad range for ") + key);
        return std::array<double, 2>{ v[0] * UM, v[1] * UM };
    };

    for (const auto& body : j.at("regions")) {
        Region r;
        r.name     = body.at("name").get<std::string>();
        r.material = body.at("material").get<std::string>();
        if (L.materials.find(r.material) == L.materials.end())
            fail("region '" + r.name + "' references unknown material '" + r.material + "'");
        r.x = range(body, "x_um");
        r.y = range(body, "y_um");
        r.z = range(body, "z_um");
        if (body.contains("power_w_cm2"))
            r.powerSurfaceWm2 = body.at("power_w_cm2").get<double>() * W_PER_CM2;
        L.regions.push_back(std::move(r));
    }
    if (L.regions.empty()) fail("no regions declared");

    // ---- boundaries -------------------------------------------------------
    for (const auto& [face, body] : j.at("boundaries").items()) {
        FaceBoundary b;
        const auto type = body.at("type").get<std::string>();
        if (type == "adiabatic")      b.type = FaceBoundary::Type::Adiabatic;
        else if (type == "robin")   { b.type = FaceBoundary::Type::Robin;
                                      b.h = body.at("h_w_m2k").get<double>();
                                      b.tInfK = body.at("t_inf_c").get<double>() + C_TO_K; }
        else if (type == "dirichlet"){ b.type = FaceBoundary::Type::Dirichlet;
                                      b.tInfK = body.at("t_c").get<double>() + C_TO_K; }
        else fail("unknown boundary type '" + type + "' on face '" + face + "'");
        L.boundaries.emplace(face, b);
    }

    // ---- validation: every cell centre must be covered --------------------
    // Cheaper and far clearer than discovering a hole mid-solve as a zero
    // conductivity. Sampling at CENTRES matches how the solver will read it.
    const auto h = L.spacing();
    for (std::size_t i = 0; i < L.cells[0]; ++i)
        for (std::size_t jj = 0; jj < L.cells[1]; ++jj)
            for (std::size_t k = 0; k < L.cells[2]; ++k)
                (void)L.materialAt((static_cast<double>(i)  + 0.5) * h[0],
                                   (static_cast<double>(jj) + 0.5) * h[1],
                                   (static_cast<double>(k)  + 0.5) * h[2]);
    return L;
}

}  // namespace physi_sim::io
