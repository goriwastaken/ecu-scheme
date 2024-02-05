
#ifndef LEHRFEMPP_PROJECTS_ECU_SCHEME_POST_PROCESSING_ECU_TOOLS_H_
#define LEHRFEMPP_PROJECTS_ECU_SCHEME_POST_PROCESSING_ECU_TOOLS_H_

#include <lf/geometry/geometry.h>
#include <lf/mesh/mesh.h>

#include <Eigen/Core>
#include <Eigen/LU>
#include <string>
#include <fstream>
#include <filesystem>
#include <memory>

namespace ecu_scheme::post_processing {

  /**
   * @brief Computes the mesh width of a triangular mesh which is defined as the maximal edge length
   * @param mesh_p underlying triangular mesh
   * @return Mesh width of the mesh
   */
  double ComputeMeshWidthTria(std::shared_ptr<const lf::mesh::Mesh> mesh_p);

  /**
   * @brief Evaluates a MeshFunction at a point specified by its global coordinates
   * @tparam MF a mesh function type
   * @param mesh_p underlying triangular mesh
   * @param mf mesh function to evaluate
   * @param global point expressed in global coordinates
   * @param tol tolerance for the search of the cell containing the point
   * @return value of the mesh function at the point
   */
  template <typename MF>
  double EvaluateMeshFunction(std::shared_ptr<const lf::mesh::Mesh> mesh_p,
                            MF mf,
                            const Eigen::Vector2d& global,
                            double tol = 10E-10){
    // Cell-wise search for the cell containing the point
    for(const lf::mesh::Entity* e : mesh_p->Entities(0)){
      LF_ASSERT_MSG(lf::base::RefEl::kTria() == e->RefEl(), "Only triangular cells are supported");
      // Compute geometric information for the cell
      const lf::geometry::Geometry* geo_ptr = e->Geometry();
      Eigen::MatrixXd corners = lf::geometry::Corners(*geo_ptr);

      // transform global coordinates to local coordinates on the cell
      Eigen::Matrix2d A;
      A << corners.col(1) - corners.col(0), corners.col(2) - corners.col(0);
      Eigen::Vector2d b;
      b << global - corners.col(0);
      Eigen::Vector2d local_coords = A.fullPivLu().solve(b);
      // Check if the point is inside the reference triangle
      if(local_coords(0) >= 0.0 - tol && local_coords(1) >= 0.0 - tol && local_coords(0) + local_coords(1) <= 1.0 + tol){
        return mf(*e, local_coords)[0];
      }
      return 0.0;
    }
  }

  /**
   * @brief Evaluates a MeshFunction at a set of points specified by their global coordinates
   * @tparam MF mesh function type
   * @param mesh_p underlying triangular mesh
   * @param mf mesh function of type MF
   * @param global set of points expressed in global coordinates
   * @param tol tolerance for the search of the cell containing the point
   * @return vector containing the values of the mesh function at the points
   */
  template <typename MF>
  std::vector<double> EvaluateMeshFunction(std::shared_ptr<const lf::mesh::Mesh> mesh_p,
                            MF mf,
                            const std::vector<Eigen::Vector2d>& global,
                            double tol = 10E-10){
    const unsigned int num_points = global.size();
    std::vector<double> result(num_points);
    std::vector<bool> computed(num_points, false);
    unsigned int counter = 0;
    for(const lf::mesh::Entity* e : mesh_p->Entities(0)){
      LF_ASSERT_MSG(lf::base::RefEl::kTria() == e->RefEl(), "Only triangular cells are supported");
      // Compute geometric information for the cell
      const lf::geometry::Geometry* geo_ptr = e->Geometry();
      Eigen::MatrixXd corners = lf::geometry::Corners(*geo_ptr);

      // transform global coordinates to local coordinates on the cell
      Eigen::Matrix2d A;
      A << corners.col(1) - corners.col(0), corners.col(2) - corners.col(0);
      Eigen::FullPivLU<Eigen::Matrix2d> lu = A.fullPivLu();

      for(unsigned int i = 0; i < num_points; ++i){
        if(!computed[i]){
          // Compute local coordinates
          Eigen::Vector2d b = global[i] - corners.col(0);
          Eigen::Vector2d local_coords = lu.solve(b);

          // Check if the point is inside the reference triangle
          if(local_coords(0) >= 0.0 - tol && local_coords(1) >= 0.0 - tol && local_coords(0) + local_coords(1) <= 1.0 + tol){
            // Evaluate the mesh function at the point
            result[i] = mf(*e, local_coords)[0];
            computed[i] = true;
            counter++;

            if(counter == num_points){
              return result;
            }
          }
        }
      }
    }
    return result;
  }

  template <typename CURVE, typename MF>
  void SampleMeshFunctionOnCurve(std::shared_ptr<const lf::mesh::Mesh> mesh_p,
                                 const CURVE& gamma,
                                 const MF& mf,
                                 unsigned int num_points,
                                 const std::string& file_name){
    // Uniform sample along gamma
    Eigen::VectorXd tau = Eigen::VectorXd::LinSpaced(num_points, 0.0, 1.0);
    std::vector<Eigen::Vector2d> sample_points(num_points);
    for(int i = 0; i < num_points; ++i){
      sample_points[i] = gamma(tau(i));
    }
    std::vector<double> result = EvaluateMeshFunction(mesh_p, mf, sample_points);

    // Generate location for output
    std::string main_dir = PROJECT_BUILD_DIR "/results";
    std::filesystem::create_directory(main_dir); //if necessary
    // Write the result to a file
    std::ofstream file;
    file.open(main_dir + "/" + file_name);
    for(int i = 0; i < num_points; ++i){
      file << tau(i) << ", " << sample_points[i](0) << ", " << sample_points[i](1) << ", " << result[i] << std::endl;
    }
    file.close();
  }



}  // namespace ecu_scheme

#endif  // LEHRFEMPP_PROJECTS_ECU_SCHEME_POST_PROCESSING_ECU_TOOLS_H_
