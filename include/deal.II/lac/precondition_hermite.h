// ---------------------------------------------------------------------
//
// Copyright (C) 2023 - 2023 by the deal.II authors
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

#ifndef dealii_precondition_hermite_h
#define dealii_precondition_hermite_h

#include <deal.II/base/config.h>

#include <deal.II/dofs/dof_handler.h>

#include <deal.II/fe/fe.h>
#include <deal.II/fe/fe_values.h>

#include <deal.II/numerics/matrix_creator.h>

#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/diagonal_matrix.h>

DEAL_II_NAMESPACE_OPEN

/**
 * @addtogroup Preconditioners
 * @{
 */

/**
 * Preconditioner class representing a rescaling of the basis vectors.
 * The rescaling is chosen so that the diagonal entries of the mass matrix
 * are all 1.
 */
template <typename Number, int dim, int spacedim = dim>
class PreconditionMassDiagonal<dim, spacedim> : public Subscriptor
{
public:
    /**
     * Default constructor that returns a preconditioner that assumes the
     * mass matrix already has 1s along the diagonal.
     */
    PreconditionMassDiagonal();

    /**
     * Constructs a preconditioner using the provided DoFHandler to evaluate
     * the values of the mass matrix along the main diagonal.
     */
    PreconditionMassDiagonal(const DoFHandler<dim, spacedim>&   dof_handler);

    /**
     * Constructs a preconditioner for the provided mass matrix. Throws an
     * exception if there is a zero or negative value somewhere on the diagonal.
     */
    PreconditionMassDiagonal(const SparseMatrix<Number>& mass_matrix);
};

DEAL_II_NAMESPACE_CLOSE

#endif