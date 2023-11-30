//
// 
//

#ifndef THESIS_MESH_BASIC_MESH_BUILDER_H_
#define THESIS_MESH_BASIC_MESH_BUILDER_H_

#include <lf/mesh/mesh.h>
#include <lf/mesh/hybrid2d/hybrid2d.h>
#include <lf/mesh/utils/utils.h>

namespace ecu_scheme::mesh {

/**
 * @brief Builds a triangular tensor product mesh on the unit square
 *
 * Some experiments discussed in the thesis require a simple tensor product mesh on the unit square
 * This class builds such a mesh with variable number of cells in each direction
 */
class BasicMeshBuilder {
 public:
  /**
   * @brief Default constructor that makes a hybrid2d mesh factory
   * Check implementation of BasicMeshBuilder::Build() for default values
   */
  BasicMeshBuilder();
  ~BasicMeshBuilder() = default;
  explicit BasicMeshBuilder(std::unique_ptr<lf::mesh::MeshFactory> mesh_factory)
        : mesh_factory_(std::move(mesh_factory)), num_cells_x_(49), num_cells_y_(49) {}

  /**
   * @brief Sets the number of cells in direction x
   * @param num_cells_x Number of cells in direction x
   */
  void SetNumCellsX(unsigned int num_cells_x) { num_cells_x_ = num_cells_x; }
  /**
   * @brief Sets the number of cells in direction y
   * @param num_cells_y number of cells in direction y
   */
  void SetNumCellsY(unsigned int num_cells_y) { num_cells_y_ = num_cells_y; }
  /**
   * @brief Builds a mesh from the mesh factory
   * @return a shared pointer to the mesh
   */
  std::shared_ptr<lf::mesh::Mesh> Build();
  /**
   * @brief Builds a tensor product mesh with an arbitrary dimension of the square
   * @param topLeftCornerX x coordinate of the top left corner of the square
   * @param topLeftCornerY y coordinate of the top left corner of the square
   * @param topRightCornerX x coordinate of the top right corner of the square
   * @param topRightCornerY y coordinate of the top right corner of the square
   * @return a shared pointer to the mesh
   */
  std::shared_ptr<lf::mesh::Mesh> Build(double topLeftCornerX, double topLeftCornerY, double topRightCornerX, double topRightCornerY);

 private:
  uint num_cells_x_{};
  uint num_cells_y_{};
  std::unique_ptr<lf::mesh::MeshFactory> mesh_factory_;
};




} // mesh

#endif //THESIS_MESH_BASIC_MESH_BUILDER_H_
