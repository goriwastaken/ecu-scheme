/**
 * @file
 * @brief Module include file for lf::base
 * @author Raffael Casagrande
 * @date   2021-01-27 10:32:40
 * @copyright MIT License
 */

#ifndef INCG986f32316282425d9be137cb399482f3
#define INCG986f32316282425d9be137cb399482f3

/**
 * @brief Contains basic functionality that is used by other parts of LehrFEM++
 */
namespace lf::base {

/** @defgroup lftypes Common Typedefs
 * @brief various integral types meant to enhance readability of the code
 */
/** @{ */
/**
 * @brief general type for variables related to size of arrays
 */
using size_type = unsigned int;
/**
 * @brief type for global index of mesh entities (nodes, edges, cells)
 */
using glb_idx_t = unsigned int;
/**
 * @brief type for local indices of sub-entities
 */
using sub_idx_t = unsigned int;
/**
 * @brief type for dimensions and co-dimensions and numbers derived from them
 */
using dim_t = unsigned int;
/**
 * @brief Index flagged as invalid
 */
const unsigned int kIdxNil = static_cast<unsigned int>(-1);
/** @} */

constexpr double kPi = 3.14159265358979323846;

}  // namespace lf::base

// public header files that make up the base library:
#include "eigen_tools.h"
#include "invalid_type_exception.h"
#include "lehrfem_info.h"
#include "lf_assert.h"
#include "lf_exception.h"
#include "narrow.h"
#include "predicate_true.h"
#include "ref_el.h"
#include "scalar_traits.h"
#include "span.h"
#include "spdlog_utils.h"
#include "timer.h"

#endif  // INCG986f32316282425d9be137cb399482f3
