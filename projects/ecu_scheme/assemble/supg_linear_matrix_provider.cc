

#include "supg_linear_matrix_provider.h"

namespace ecu_scheme::assemble {
  double Diameter(const lf::mesh::Entity &entity){
    const lf::geometry::Geometry* geo_p = entity.Geometry();
    Eigen::MatrixXd corners = lf::geometry::Corners(*geo_p);

    switch (entity.RefEl()) {
      case lf::base::RefEl::kTria(): {
        // Diameter of a triangle corresponds to the longest edge
        Eigen::Vector2d e0 = corners.col(1) - corners.col(0);
        Eigen::Vector2d e1 = corners.col(2) - corners.col(1);
        Eigen::Vector2d e2 = corners.col(0) - corners.col(1);
        return std::max(e0.norm(), std::max(e1.norm(), e2.norm()));
      }
      case lf::base::RefEl::kQuad(): {
        // Diameter of a (convex) quadrilateral corresponds to the longer diagonal
        Eigen::Vector2d d0 = corners.col(2) - corners.col(0);
        Eigen::Vector2d d1 = corners.col(3) - corners.col(1);
        return std::max(d0.norm(), d1.norm());
      }
      default: {
        LF_ASSERT_MSG(false,
                      "Diameter not available for " << entity.RefEl().ToString());
        return 0.0;
      }
    }
  }

  double Delta(const lf::mesh::Entity& entity, double eps, Eigen::Vector2d v) {
    double const h = ecu_scheme::assemble::Diameter(entity);
    double const v_norm = v.lpNorm<Eigen::Infinity>();

    if (v_norm * h / (2 * eps) <= 1.0) {
      return h * h / eps;
    }
    return h;

  }
} // namespace ecu_scheme
