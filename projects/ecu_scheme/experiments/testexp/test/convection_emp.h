#ifndef CONVECTION_EMP_H
#define CONVECTION_EMP_H

/**
 * @file convection_emp.h
 * @brief EMP for a convection term based on linear FE and the trapezoidal rule
 * @author Philippe Peter
 * @date July 2021
 * @copyright Developed at SAM, ETH Zurich
 */

#include <lf/base/base.h>
#include <lf/geometry/geometry.h>
#include <lf/mesh/mesh.h>

#include <Eigen/Core>

namespace ConvectionDiffusion {

/**
 * @headerfile convection_emp.h
 * @brief Computes the local matrices for the convection term based on linear
 * finite elements and the trapezoidal rule (standard Galerkin approach).
 *
 * @tparam FUNCTOR function that defines the vector valued velocity
 * coefficient v.
 */
template <typename FUNCTOR>
class ConvectionElementMatrixProvider {
 public:
  /**
   * @brief
   * @param v functor for the velocity field
   */
  explicit ConvectionElementMatrixProvider(FUNCTOR v) : v_(v) {}

  /**
   * @brief main routine for the computation of element matrices
   * @param enttity reference to the TRIANGULAR cell for which the element
   * matrix should be computed.
   * @return a 3x3 matrix containing the element matrix.
   */
  Eigen::Matrix3d Eval(const lf::mesh::Entity &entity);

  /** @brief Default implementation: all cells are active */
  bool isActive(const lf::mesh::Entity & /*entity*/) const { return true; }

 private:
  FUNCTOR v_;  // functor for the velocity field.
};

template <typename FUNCTOR>
Eigen::Matrix3d ConvectionElementMatrixProvider<FUNCTOR>::Eval(
    const lf::mesh::Entity &entity) {
  LF_ASSERT_MSG(lf::base::RefEl::kTria() == entity.RefEl(),
                "Function only defined for triangular cells");

  const lf::geometry::Geometry *geo_ptr = entity.Geometry();
  Eigen::Matrix3d loc_mat;

  const Eigen::MatrixXd corners = lf::geometry::Corners(*geo_ptr);

  // calculate the gradients of the basis functions.
  // See \lref{cpp:gradbarycoords}, \lref{mc:ElementMatrixLaplLFE} for details.
  Eigen::Matrix3d grad_helper;
  grad_helper.col(0) = Eigen::Vector3d::Ones();
  grad_helper.rightCols(2) = corners.transpose();
  // Matrix with gradients of the local shape functions in its columns
  const Eigen::MatrixXd grad_basis = grad_helper.inverse().bottomRows(2);

  // evaluate velocity field at the vertices:
  Eigen::MatrixXd velocities(2, 3);
  velocities << v_(corners.col(0)), v_(corners.col(1)), v_(corners.col(2));

  // compute local matrix using local trapezoidal rule:
  loc_mat = lf::geometry::Volume(*geo_ptr) / 3.0 * velocities.transpose() *
            grad_basis;

  return loc_mat;
}

}  // namespace ConvectionDiffusion

namespace ConvectionDiffusionQuadratic{

  template <typename FUNCTOR>
  class ConvectionQuadraticElementMatrixProvider {
   public:
    /**
     * @brief
     * @param v functor for the velocity field
     */
    ConvectionQuadraticElementMatrixProvider(std::shared_ptr<const lf::uscalfe::UniformScalarFESpace<double>> fe_space
                                             ,FUNCTOR v);



    /**
     * @brief main routine for the computation of element matrices
     * @param enttity reference to the TRIANGULAR cell for which the element
     * matrix should be computed.
     * @return a 3x3 matrix containing the element matrix.
     */
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> Eval(const lf::mesh::Entity &entity);

    /** @brief Default implementation: all cells are active */
    bool isActive(const lf::mesh::Entity & /*entity*/) const { return true; }

   private:
    FUNCTOR v_;  // functor for the velocity field.
    std::array<lf::uscalfe::PrecomputedScalarReferenceFiniteElement<double>, 5> fe_precomp_;
  };

  template <typename FUNCTOR>
  ConvectionQuadraticElementMatrixProvider<FUNCTOR>::ConvectionQuadraticElementMatrixProvider(std::shared_ptr<const lf::uscalfe::UniformScalarFESpace<
      double>> fe_space, FUNCTOR v) : v_(std::move(v)), fe_precomp_() {

      for(auto ref_el : {lf::base::RefEl::kTria(), lf::base::RefEl::kQuad()}){
        auto fe = fe_space->ShapeFunctionLayout(ref_el);
        if(fe != nullptr){
          fe_precomp_[ref_el.Id()] =
              lf::uscalfe::PrecomputedScalarReferenceFiniteElement(fe, lf::quad::make_QuadRule(ref_el, 2*fe->Degree()));
        }
      }

      }

  template <typename FUNCTOR>
  Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> ConvectionQuadraticElementMatrixProvider<FUNCTOR>::Eval(
      const lf::mesh::Entity &cell) {
    LF_ASSERT_MSG(lf::base::RefEl::kTria() == cell.RefEl(),
                  "Function only defined for triangular cells");
    //compute element matrix for quadratic lagrangian FE space for a single triangle
    const lf::base::RefEl ref_el = cell.RefEl();
    lf::uscalfe::PrecomputedScalarReferenceFiniteElement<double> &pfe = fe_precomp_[ref_el.Id()];

    if (!pfe.isInitialized()) {
      // Accident: cell is of a type not covered by finite element
      // specifications or there is no quadrature rule available for this
      // reference element type
      std::stringstream temp;
      temp << "No local shape function information or no quadrature rule for "
              "reference element type "
           << ref_el;
      throw lf::base::LfException(temp.str());
    }

    // Query the shape of the cell
    const lf::geometry::Geometry *geo_ptr = cell.Geometry();
    LF_ASSERT_MSG(geo_ptr != nullptr, "Invalid geometry!");
    LF_ASSERT_MSG((geo_ptr->DimLocal() == 2),
                  "Only 2D implementation available!");

    // Physical dimension of the cell (must be 2)
    const lf::base::dim_t world_dim = geo_ptr->DimGlobal();
    LF_ASSERT_MSG_CONSTEXPR(world_dim == 2, "Only available for flat domains");

    // Gram determinant at quadrature points
    const Eigen::VectorXd determinants(
        geo_ptr->IntegrationElement(pfe.Qr().Points()));
    LF_ASSERT_MSG(
        determinants.size() == pfe.Qr().NumPoints(),
        "Mismatch " << determinants.size() << " <-> " << pfe.Qr().NumPoints());
    // Fetch the transformation matrices for the gradients
    const Eigen::MatrixXd JinvT(
        geo_ptr->JacobianInverseGramian(pfe.Qr().Points()));
    LF_ASSERT_MSG(
        JinvT.cols() == 2 * pfe.Qr().NumPoints(),
        "Mismatch " << JinvT.cols() << " <-> " << 2 * pfe.Qr().NumPoints());
    LF_ASSERT_MSG(JinvT.rows() == world_dim,
                  "Mismatch " << JinvT.rows() << " <-> " << world_dim);

    // Get the velocity vectors at quadrature points in the cell
    auto veloval = v_(cell, pfe.Qr().Points());

    // Element matrix
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> mat(pfe.NumRefShapeFunctions(), pfe.NumRefShapeFunctions());
    mat.setZero();
    //LF_ASSERT_MSG(mat.cols() == 6, "Wrong size of element matrix");

    for(lf::base::size_type k = 0; k< pfe.Qr().NumPoints(); ++k){
      const double w = pfe.Qr().Weights()[k] * determinants[k];
      const auto trf_grad(
          JinvT.block(0, 2*static_cast<Eigen::Index>(k), world_dim, 2)*
          pfe.PrecompGradientsReferenceShapeFunctions()
          .block(0, 2*k, mat.rows(), 2).transpose());

      const auto velo_times_trf_grad(veloval[k].transpose()*trf_grad);
      mat += w * (pfe.PrecompReferenceShapeFunctions().col(k) * velo_times_trf_grad);
    }

    return mat;
  }

} // namespace ConvectioinDiffusionQuadratic

#endif  // CONVECTIOIN_EMP_H