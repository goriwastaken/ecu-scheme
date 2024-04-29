
#include "ecu_tools.h"

namespace ecu_scheme::post_processing {

double ComputeMeshWidthTria(std::shared_ptr<const lf::mesh::Mesh> mesh_p) {
  double max_width = 0.0;
  for (const lf::mesh::Entity* cell : mesh_p->Entities(0)) {
    LF_ASSERT_MSG(lf::base::RefEl::kTria() == cell->RefEl(),
                  "Only triangular cells are supported");
    // Maximal mesh width of mesh corresponds to the maximal edge length
    for (const lf::mesh::Entity* edge : cell->SubEntities(1)) {
      const double edge_length = lf::geometry::Volume(*edge->Geometry());
      if (edge_length > max_width) {
        max_width = edge_length;
      }
    }
  }
  return max_width;
}
}  // namespace ecu_scheme::post_processing