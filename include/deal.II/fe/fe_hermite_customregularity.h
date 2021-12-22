// ---------------------------------------------------------------------
//
// Copyright (C) 2021 - 2022 by the deal.II authors
//
// This file is part of the deal.II library.
//
// The deal.II library is free software; you can use it, redistribute
// it, and/or modify it under the terms of the GNU Lesser General
// Public License as published by the Free Software Foundation; either
// version 2.1 of the License, or (at your option) any later version.
// The full text of the license can be found in the file LICENSE.md at
// the top level directory of deal.II.
//
// ---------------------------------------------------------------------

#ifndef dealii_fe_cont_hermite_customregularity
#define dealii_fe_cont_hermite_customregularity

#include <deal.II/base/config.h>

#include <deal.II/base/polynomials_hermite_customregularity.h>
#include <deal.II/base/tensor_product_polynomials.h>
#include <deal.II/base/thread_management.h>

#include <deal.II/dofs/dof_handler.h>

#include <deal.II/fe/fe_poly.h>
#include <deal.II/fe/mapping_hermite.h>

#include <deal.II/numerics/vector_tools_project.h>

#include <string>
#include <vector>

DEAL_II_NAMESPACE_OPEN



/**
 * Header file for constructing a Hermite basis of custom regularity 
 * elements. These are defined by adding Lagrange-style interpolation 
 * nodes inside the elements. The number of nodes must be non-zero,
 * otherwise an error is returned. For a Hermite basis with zero nodes the 
 * standard maximum regularity basis (FE_Hermite) should be used instead.
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
 * @code
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
 * @endcode
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
 * This class is only for custom regularity below the maximum. To use maximum
 * regularity see FE_Hermite.
 */

template <int dim, int spacedim = dim>
class FE_CustomHermite : public FE_Poly<dim, spacedim>
{
public:
  /**
   * Constructors
   */
  FE_CustomHermite<dim, spacedim>(const unsigned int regularity, const unsigned int nodes);

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
   */
  /*
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
  
  virtual void
  fill_fe_values(
    const typename Triangulation<dim, spacedim>::cell_iterator &cell,
    const CellSimilarity::Similarity                            cell_similarity,
    const Quadrature<dim> &                                     quadrature,
    const Mapping<dim, spacedim> &                              mapping,
    const typename Mapping<dim, spacedim>::InternalDataBase &mapping_internal,
    const dealii::internal::FEValuesImplementation::MappingRelatedData<dim,
                                                                       spacedim>
      &                                                            mapping_data,
    const typename FiniteElement<dim, spacedim>::InternalDataBase &fe_internal,
    dealii::internal::FEValuesImplementation::FiniteElementRelatedData<dim,
                                                                       spacedim>
      &output_data) const override;
      
  using FiniteElement<dim, spacedim>::fill_fe_face_values;
  
  virtual void
  fill_fe_face_values(
    const typename Triangulation<dim, spacedim>::cell_iterator &cell,
    const unsigned int                                          face_no,
    const hp::QCollection<dim - 1> &                                 quadrature,
    const Mapping<dim, spacedim> &                              mapping,
    const typename Mapping<dim, spacedim>::InternalDataBase &   mapping_internal,
    const dealii::internal::FEValuesImplementation::MappingRelatedData<dim,
                                                                     spacedim>
      &                                                            mapping_data,
    const typename FiniteElement<dim, spacedim>::InternalDataBase &fe_internal,
    dealii::internal::FEValuesImplementation::FiniteElementRelatedData<dim,
                                                                     spacedim>
    &output_data) const override;
      
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
  
  /*
  void 
  initialize_quad_dof_index_permutation();
  */

  struct Implementation;
  friend struct FE_CustomHermite<dim, spacedim>::Implementation;
  
  /*
  virtual std::unique_ptr<
  typename FiniteElement<dim, spacedim>::InternalDataBase>
  get_data(
    const UpdateFlags update_flags,
    const Mapping<dim, spacedim> & / *mapping* /,
    const Quadrature<dim> &quadrature,
    dealii::internal::FEValuesImplementation::FiniteElementRelatedData<dim,
                                                                       spacedim>
      &/ * output_data* /) const override
  {
    // generate a new data object and
    // initialize some fields
    std::unique_ptr<typename FiniteElement<dim, spacedim>::InternalDataBase>
          data_ptr   = std::make_unique<typename FE_Poly<dim, spacedim>::InternalData>();
    auto &data       = dynamic_cast<typename FE_Poly<dim, spacedim>::InternalData &>(*data_ptr);
    data.update_each = this->requires_update_flags(update_flags);

    const unsigned int n_q_points = quadrature.size();

    // initialize some scratch arrays. we need them for the underlying
    // polynomial to put the values and derivatives of shape functions
    // to put there, depending on what the user requested
    std::vector<double> values(
      update_flags & update_values ? this->dofs_per_cell : 0);
    std::vector<Tensor<1, dim>> grads(
      update_flags & update_gradients ? this->dofs_per_cell : 0);
    std::vector<Tensor<2, dim>> grad_grads(
      update_flags & update_hessians ? this->dofs_per_cell : 0);
    std::vector<Tensor<3, dim>> third_derivatives(
      update_flags & update_3rd_derivatives ? this->dofs_per_cell : 0);
    std::vector<Tensor<4, dim>>
      fourth_derivatives; // won't be needed, so leave empty

    // now also initialize fields the fields of this class's own
    // temporary storage, depending on what we need for the given
    // update flags.
    //
    // there is one exception from the rule: if we are dealing with
    // cells (i.e., if this function is not called via
    // get_(sub)face_data()), then we can already store things in the
    // final location where FEValues::reinit() later wants to see
    // things. we then don't need the intermediate space. we determine
    // whether we are on a cell by asking whether the number of
    // elements in the output array equals the number of quadrature
    // points (yes, it's a cell) or not (because in that case the
    // number of quadrature points we use here equals the number of
    // quadrature points summed over *all* faces or subfaces, whereas
    // the number of output slots equals the number of quadrature
    // points on only *one* face)
    if (update_flags & update_values)
      data.shape_values.reinit(this->dofs_per_cell, n_q_points);

    if (update_flags & update_gradients)
      data.shape_gradients.reinit(this->dofs_per_cell, n_q_points);

    if (update_flags & update_hessians)
      data.shape_hessians.reinit(this->dofs_per_cell, n_q_points);

    if (update_flags & update_3rd_derivatives)
      data.shape_3rd_derivatives.reinit(this->dofs_per_cell, n_q_points);

    // next already fill those fields of which we have information by
    // now. note that the shape gradients are only those on the unit
    // cell, and need to be transformed when visiting an actual cell
    if (update_flags & (update_values | update_gradients | update_hessians |
                        update_3rd_derivatives))
      for (unsigned int i = 0; i < n_q_points; ++i)
        {
          this->poly_space->evaluate(quadrature.point(i),
                              values,
                              grads,
                              grad_grads,
                              third_derivatives,
                              fourth_derivatives);

          // for Hermite everything needs to be transformed,
          // so we write them into our scratch space and only later
          // copy stuff into where FEValues wants it
          if (update_flags & update_values)
            for (unsigned int k = 0; k < this->dofs_per_cell; ++k)
              data.shape_values[k][i] = values[k];

          if (update_flags & update_gradients)
            for (unsigned int k = 0; k < this->dofs_per_cell; ++k)
              data.shape_gradients[k][i] = grads[k];

          if (update_flags & update_hessians)
            for (unsigned int k = 0; k < this->dofs_per_cell; ++k)
              data.shape_hessians[k][i] = grad_grads[k];

          if (update_flags & update_3rd_derivatives)
            for (unsigned int k = 0; k < this->dofs_per_cell; ++k)
              data.shape_3rd_derivatives[k][i] = third_derivatives[k];
        }
    return data_ptr;
  }
  */
  
  virtual std::unique_ptr<
  typename FiniteElement<dim, spacedim>::InternalDataBase>
  get_data(
    const UpdateFlags             update_flags,
    const Mapping<dim, spacedim> &mapping,
    const Quadrature<dim> &       quadrature,
    dealii::internal::FEValuesImplementation::FiniteElementRelatedData<dim,
                                                                       spacedim>
      &output_data) const override
  {
    std::unique_ptr<typename FiniteElement<dim, spacedim>::InternalDataBase>
          data_ptr   = FE_Poly<dim, spacedim>::get_data(update_flags, mapping, quadrature, output_data);
    auto &data       = dynamic_cast<typename FE_Poly<dim, spacedim>::InternalData &>(*data_ptr);
    const unsigned int n_q_points = quadrature.size();
    if ((update_flags & update_values) &&
        ((output_data.shape_values.n_rows() > 0) &&
         (output_data.shape_values.n_cols() == n_q_points)))
      data.shape_values = output_data.shape_values;
    return data_ptr;
  }

private:
  mutable Threads::Mutex mutex;
  unsigned int           regularity;
  unsigned int           nodes;
  
public:
  inline unsigned int
  get_regularity() const
  {return this->regularity;};
  
  inline unsigned int
  n_nodes() const
  {return this->nodes;};
};



namespace VectorTools
{
    enum HermiteBoundaryType {
            hermite_dirichlet,
            hermite_neumann,
            hermite_2nd_derivative,
            hermite_robin,
            hermite_other_combined
    };
    
    
    template <int dim, int spacedim = dim, typename Number = double>
    void
    project_boundary_values(const MappingHermite<dim, spacedim> &                                  mapping_h,
                            const DoFHandler<dim, spacedim> &                                      dof_handler,
                            const std::map<types::boundary_id, const Function<spacedim, Number>*> &boundary_functions,
                            const Quadrature<dim - 1> &                                            quadrature,
                            const HermiteBoundaryType                                              projection_mode,
                            std::map<types::global_dof_index, Number> &                            boundary_values,
                            std::vector<unsigned int>                                              component_mapping = {});
    
    //The following does exactly the same as the above, but uses hermite_dirichlet. This is to match existing function calls
    template <int dim, int spacedim = dim, typename Number = double>
    void
    project_boundary_values(const MappingHermite<dim, spacedim> &                                  mapping_h,
                            const DoFHandler<dim, spacedim> &                                      dof_handler,
                            const std::map<types::boundary_id, const Function<spacedim, Number>*> &boundary_functions,
                            const Quadrature<dim - 1> &                                            quadrature,
                            std::map<types::global_dof_index, Number> &                            boundary_values,
                            std::vector<unsigned int>                                              component_mapping = {});
    
    template <int dim, int spacedim = dim, typename Number = double>
    void
    project_boundary_values(const MappingHermite<dim, spacedim> &                                  mapping_h,
                            const DoFHandler<dim, spacedim> &                                      dof_handler,
                            const std::map<types::boundary_id, const Function<spacedim, Number>*> &boundary_functions,
                            const Quadrature<dim - 1> &                                            quadrature,
                            const HermiteBoundaryType                                              projection_mode,
                            AffineConstraints<Number>                                              constraints,
                            std::vector<unsigned int>                                              component_mapping = {});
    
    //Same as above, but with hermite_dirichlet set
    template <int dim, int spacedim = dim, typename Number = double>
    void
    project_boundary_values(const MappingHermite<dim, spacedim> &                                  mapping_h,
                            const DoFHandler<dim, spacedim> &                                      dof_handler,
                            const std::map<types::boundary_id, const Function<spacedim, Number>*> &boundary_functions,
                            const Quadrature<dim - 1> &                                            quadrature,
                            AffineConstraints<Number>                                              constraints,
                            std::vector<unsigned int>                                              component_mapping = {});
    
    //Overwrite the following function in the case of Hermite elements to account for poorer conditioning
    using VectorTools::project;
    
    template <int dim, typename VectorType, int spacedim> void
    project(const MappingHermite<dim, spacedim> &                     mapping,
            const DoFHandler<dim, spacedim> &                         dof,
            const AffineConstraints<typename VectorType::value_type> &constraints,
            const Quadrature<dim> &                                   quadrature,
            const Function<spacedim, typename VectorType::value_type> &function,
            VectorType &                                               vec,
            const bool                 enforce_zero_boundary     = false,
            const Quadrature<dim - 1> &q_boundary                = (dim > 1 ?
                                                       QGauss<dim - 1>(2) :
                                                       Quadrature<dim - 1>(0)),
            const bool                 project_to_boundary_first = false);
    
}

DEAL_II_NAMESPACE_CLOSE

#endif
