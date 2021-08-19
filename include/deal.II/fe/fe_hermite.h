
#ifndef dealii_fe_cont_hermite
#define dealii_fe_cont_hermite

#define HERMITE_CUSTOM_FE_CLASS 0

#include <deal.II/base/config.h>

#include <deal.II/base/polynomials_hermite.h>
#include <deal.II/base/tensor_product_polynomials.h>
#include <deal.II/base/thread_management.h>

#include <deal.II/fe/fe_poly.h>

#include <string>
#include <vector>


DEAL_II_NAMESPACE_OPEN

/**
 * Header file for constructing a Hermite basis. Maximum regularity elements
 * are always of odd polynomial degree, have regularity $q = (p-1)/2$ and are
 * defined up to degree $p=13$. Custom regularity elements are defined by
 * adding Lagrange-style interplation nodes inside the elements.
 *
 * For maximum regularity elements each node has (q+1)^d degrees of freedom
 * (d.o.fs) assigned to it, corresponding to the different combinations of
 * directional derivatives up to the required regularity at that node. DoFs
 * at each node are not consecutive in either the global or local ordering,
 * due to the tensor product construction of the basis. The ordering is
 * determined by the direction of the derivative each function corresponds to;
 * first by x-derivatives, then y, then z. Locally over each element the
 * d.o.fs are ordered similarly. See below for the local ordering for
 * regularity 1 (FE_MaxHermite(1)):
 *
 *       (40,41,44,45)__(42,43,46,47)          (40,41,44,45)__(42,43,46,47)
 *       (56,57,60,61)  (58,59,62,63)          (56,57,60,61)  (58,59,62,63)
 *          /  |              |                    /           /  |
 *         /   |              |                   /           /   |
 *        /    |              |                  /           /    |
 *(32,33,36,37)|              |         (32,33,36,37)__(34,35,38,39)
 *(48,49,52,53)|              |         (48,49,52,53)  (50,51,54,55)
 *       |     |              |                |            |     |
 *       |( 8,9,12,13 )__(10,11,14,15)         |            |(10,11,14,15)
 *       |(24,25,28,29)  (26,27,30,31)         |            |(26,27,30,31)
 *       |    /              /                 |            |    /
 *       |   /              /                  |            |   /
 *       |  /              /                   |            |  /
 *  (  0,1,4,5  )___(  2,3,6,7  )      (  0,1,4,5  )___(  2,3,6,7  )
 *  (16,17,20,21)   (18,19,22,23)      (16,17,20,21)   (18,19,22,23)

 *
 * For custom regularity elements extra degrees of freedom are included at
 * interpolation points along the element edges, faces and in the element
 * interior. Edge nodes will have (q+1)^(d-1) d.o.fs, face nodes (q+1)^(d-2)
 * etc.  This will lead to a total of (2q+n+2)^d basis functions that are
 * non-zero on any given element, where n is the number of interpolation nodes
 * added in a given direction. d.o.fs are ordered by derivatives in the same
 * way as before, with derivatives parallel to the edge or face an
 * interpolation node is on disregarded. See below for the local ordering over
 * an element for regularity 1 with one interpolation node
 * (FE_CustomHermite(1,1)):
 *
 *
 *                 (90,91,95,96)  __  (92,97) ___  (93,94,98,99)
 *               (115,116,120,121)   (117,122)   (118,119,123,124)
 *                      /|                               |
 *                     / |                               |
 *                    /  |                               |
 *                   /   |                               |
 *               (85,86) |                               |
 *              (110,111)|                               |
 *                /      |                               |
 *               /       |                               |
 *              /  (65,66,70,71)      (67,72)      (68,69,73,74)
 *             /         |                               |
 *       (75,76,80,81)   |                               |
 *     (100,101,105,106) |                               |
 *            |          |                               |
 *            |          |                               |
 *            |  (60,61) |       (62)                    |
 *            |          |                               |
 *            |          |                               |
 *            |          |                               |
 *            |    (15,16,20,21)______(17,22)______(18,19,23,24)
 *            |    (40,41,45,46)      (42,47)      (43,44,48,49)
 *      (50,51,55,56)   /                              /
 *            |        /                              /
 *            |       /                              /
 *            |      /                              /
 *            |  (10,11)         (12)           (13,14)
 *            |  (35,36)         (37)           (38,39)
 *            |   /                              /
 *            |  /                              /
 *            | /                              /
 *            |/                              /
 *      (  0,1,5,6  )______( 2,7 )______(  3,4,8,9  )
 *      (25,26,30,31)      (27,32)      (28,29,33,34)
 *
 *
 *
 *                 (90,91,95,96)  __ (92,97) ___  (93,94,98,99)
 *               (115,116,120,121)  (117,122)   (118,119,123,124)
 *                      /                              /|
 *                     /                              / |
 *                    /                              /  |
 *                   /                              /   |
 *               (85,86)         (87)           (88,89) |
 *              (110,111)       (112)          (113,114)|
 *                /                              /      |
 *               /                              /       |
 *              /                              /  (68,69,73,74)
 *             /                              /         |
 *      (75,76,80,81)  ___ (77,82) __  (78,79,83,84)    |
 *    (100,101,105,106)   (102,107)  (103,104,108,109)  |
 *            |                              |          |
 *            |                              |          |
 *            |                              | (63,64)  |
 *            |                              |          |
 *            |                              |          |
 *            |                              |          |
 *            |                              |    (18,19,23,24)
 *            |                              |    (43,44,48,49)
 *      (50,51,55,56)      (52,57)     (53,54,58,59)   /
 *            |                              |        /
 *            |                              |       /
 *            |                              |      /
 *            |                              |  (13,14)
 *            |                              |  (38,39)
 *            |                              |   /
 *            |                              |  /
 *            |                              | /
 *            |                              |/
 *      (  0,1,5,6  )______( 2,7 )______(  3,4,8,9  )
 *      (25,26,30,31)      (27,32)      (28,29,33,34)
 *
 *
 * DoF 62 lies at the centre of the cell in the above diagram.
 *
 * The interpolation points used lie at the Chebyshev Gauss-Lobatto points in
 * each direction. This distribution was chosen due to being easy to calculate
 * (an explicit formula exists) and to avoid the issue of the Runge phenomenom
 * as the number of nodes becomes larger.
 *
 * Note that while the number of functions defined on each cell appears large,
 * due to the increased regularity constraints many of these functions are
 * shared between elements.
 */

/*
 * Use one class for both maximum regularity and custom regularity. To take
 * advantage of the easier math for the max regularity, define two sets of
 * functions differentiated by the number of input arguments.
 */

template <int dim, int spacedim = dim>
class FE_Hermite : public FE_Poly<dim, spacedim>
{
public:
  /*
   * Constructors
   */
  FE_Hermite<dim, spacedim>(const unsigned int regularity);
#if HERMITE_CUSTOM_FE_CLASS
  FE_Hermite<dim, spacedim>(const unsigned int regularity, const unsigned int nodes);
#endif

  /*
   * Matrix functions, comparable to FE_Q_Base implementations
   */
  /*
  virtual void
  get_interpolation_matrix(const FiniteElement<dim, spacedim> &source,
                         FullMatrix<double> &matrix) const override;

  virtual void
  get_face_interpolation_matrix(const FiniteElement<dim, spacedim> &source,
                              FullMatrix<double> &matrix) const override;

  virtual void
  get_subface_interpolation_matrix(const FiniteElement<dim, spacedim> &source,
                                 const unsigned int                  subface,
                                 FullMatrix<double> &matrix) const override;

  virtual bool
  has_support_on_face(const unsigned int shape_index,
                    const unsigned int face_index) const override;

  virtual const FullMatrix<double> &
  get_restriction_matrix(const unsigned int child,
                         const RefinementCase<dim> &refinement_case =
                         RefinementCase<dim>::isotropic_refinement) const
  override;

  virtual const FullMatrix<double> &
  get_prolongation_matrix(const unsigned int child,
                          const RefinementCase<dim> &refinement_case =
                          RefinementCase<dim>::isotropic_refinement) const
  override;

  virtual unsigned int
  face_to_cell_index(const unsigned int face_dof_index,
                     const unsigned int face,
                     const bool         face_orientation = true,
                     const bool         face_flip        = false,
                     const bool         face_rotation = false) const override;

  virtual std::pair<Table<2, bool>, std::vector<unsigned int>>
  get_constant_modes() const override;                                  // Should be quick to implement 
  */
  /*
   * hp functions

  virtual bool
  hp_constraints_are_implemented() const override;

  virtual std::vector<std::pair<unsigned int, unsigned int>>
  hp_vertex_dof_identities(const FiniteElement<dim, spacedim> &fe_other) const
  override;

  virtual std::vector<std::pair<unsigned int, unsigned int>>
  hp_line_dof_identities(const FiniteElement<dim, spacedim> &fe_other) const
  override;

  virtual std::vector<std::pair<unsigned int, unsigned int>>
  hp_quad_dof_identities(const FiniteElement<dim, spacedim> &fe_other) const
  override;
  */
  /*
   * Other functions
   */
  virtual std::string
  get_name() const override;

  virtual std::unique_ptr<FiniteElement<dim, spacedim>>
  clone() const override;
  /*
  virtual std::size_t
  memory_consumption() const override;

  virtual void
  convert_generalized_support_point_values_to_dof_values(
  const std::vector<Vector<double>> &support_point_values,
  std::vector<double> &              nodal_values) const override;

  virtual FiniteElementDomination::Domination
  compare_for_domination(const FiniteElement<dim, spacedim> &fe_other,
                       const unsigned int codim = 0) const override final;
  */
protected:
  void
  initialize_constraints();
  // void initialize_quad_dof_index_permutation();

  struct Implementation;
  friend struct FE_Hermite<dim, spacedim>::Implementation;

private:
  //mutable Threads::Mutex mutex;
  unsigned int           regularity;
  unsigned int           nodes;
};



DEAL_II_NAMESPACE_CLOSE

#endif
