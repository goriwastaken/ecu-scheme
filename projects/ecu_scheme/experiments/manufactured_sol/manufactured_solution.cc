
#include "manufactured_solution.h"

namespace ecu_scheme::experiments {

void EnforceBoundaryConditions(
    const std::shared_ptr<lf::uscalfe::UniformScalarFESpace<double>>& fe_space,
    lf::assemble::COOMatrix<double>& A, Eigen::VectorXd& phi,
    std::function<double(const Eigen::Matrix<double, 2, 1, 0>&)> dirichlet) {
  lf::mesh::utils::MeshFunctionGlobal<decltype(dirichlet)> mf_g_dirichlet{
      dirichlet};
  lf::mesh::utils::AllCodimMeshDataSet<bool> bd_flags(fe_space->Mesh(), false);
  // set a fixed epsilon value for the geometric test involving double
  // comparison
  const double kTol = 1e-8;
  // Loop over all edges
  for (const auto& edge : fe_space->Mesh()->Entities(1)) {
    LF_ASSERT_MSG(edge->RefEl() == lf::base::RefEl::kSegment(),
                  "Entity should be an edge");
    const lf::geometry::Geometry* geo_ptr = edge->Geometry();
    const Eigen::MatrixXd corners = lf::geometry::Corners(*geo_ptr);
    // Check if the edge lies on $\Gamma_{\mathrm{in}}$  (geometric test)
    if ((corners(0, 0) + corners(0, 1)) / 2. < /*> 1. -*/ kTol ||
        (corners(1, 0) + corners(1, 1)) / 2. < kTol) {
      // Add the edge to the flagged entities
      bd_flags(*edge) = true;
    }
  }
  // Loop over all Points
  for (const auto& point : fe_space->Mesh()->Entities(2)) {
    LF_ASSERT_MSG(point->RefEl() == lf::base::RefEl::kPoint(),
                  "Entity should be an edge");
    const lf::geometry::Geometry* geo_ptr = point->Geometry();
    const Eigen::VectorXd coords = lf::geometry::Corners(*geo_ptr);
    // Check if the node lies on  $\Gamma_{\mathrm{in}}$ (geometric test)
    if (coords(0) < /*> 1. -*/ kTol || coords(1) < kTol) {
      // Add the edge to the flagged entities
      bd_flags(*point) = true;
    }
  }
  auto flag_values{lf::fe::InitEssentialConditionFromFunction(
      *fe_space, bd_flags, mf_g_dirichlet)};

  lf::assemble::FixFlaggedSolutionCompAlt<double>(
      [&flag_values](lf::assemble::glb_idx_t dof_idx) {
        return flag_values[dof_idx];
      },
      A, phi);
}

}  // namespace ecu_scheme::experiments