// ---------------------------------------------------------------------
//
// Copyright (C) 2012 - 2018 by the deal.II authors
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


#include <deal.II/fe/block_mask.h>

#include <iostream>


DEAL_II_NAMESPACE_OPEN

std::ostream &
operator<<(std::ostream &out, const BlockMask &mask)
{
  if (mask.block_mask.empty())
    out << "[all blocks selected]";
  else
    {
      out << '[';
      for (unsigned int i = 0; i < mask.block_mask.size(); ++i)
        {
          out << (mask.block_mask[i] ? "true" : "false");
          if (i != mask.block_mask.size() - 1)
            out << ',';
        }
      out << ']';
    }

  return out;
}



std::size_t
BlockMask::memory_consumption() const
{
  return sizeof(*this) + MemoryConsumption::memory_consumption(block_mask);
}


DEAL_II_NAMESPACE_CLOSE
