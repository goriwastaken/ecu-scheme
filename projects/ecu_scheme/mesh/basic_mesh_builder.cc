//
//
//

#include "basic_mesh_builder.h"

namespace ecu_scheme::mesh {

/**
 * @brief Default constructor that makes a hybrid2d mesh factory
 */
BasicMeshBuilder::BasicMeshBuilder() {
  mesh_factory_ = std::make_unique<lf::mesh::hybrid2d::MeshFactory>(2);
  SetNumCellsX(49);
  SetNumCellsY(49);
}

std::shared_ptr<lf::mesh::Mesh> BasicMeshBuilder::Build() {
  const uint num_cells_x = num_cells_x_;
  const uint num_cells_y = num_cells_y_;

  std::shared_ptr<lf::mesh::Mesh> mesh_p;

  // create a builder based on the mesh factory
  lf::mesh::utils::TPTriagMeshBuilder builder(std::move(mesh_factory_));

  builder.setBottomLeftCorner(Eigen::Vector2d{0.0, 0.0})
      .setTopRightCorner(Eigen::Vector2d{1.0, 1.0})
      .setNumXCells(num_cells_x)
      .setNumYCells(num_cells_y);

  mesh_p = builder.Build();

  return mesh_p;
}

std::shared_ptr<lf::mesh::Mesh> BasicMeshBuilder::Build(
    double topLeftCornerX, double topLeftCornerY, double topRightCornerX,
    double topRightCornerY) {
  const uint num_cells_x = num_cells_x_;
  const uint num_cells_y = num_cells_y_;

  std::shared_ptr<lf::mesh::Mesh> mesh_p;

  // create a builder based on the mesh factory
  lf::mesh::utils::TPTriagMeshBuilder builder(std::move(mesh_factory_));

  builder.setBottomLeftCorner(Eigen::Vector2d{topLeftCornerX, topLeftCornerY})
      .setTopRightCorner(Eigen::Vector2d{topRightCornerX, topRightCornerY})
      .setNumXCells(num_cells_x)
      .setNumYCells(num_cells_y);

  mesh_p = builder.Build();

  return mesh_p;
}

}  // namespace ecu_scheme::mesh