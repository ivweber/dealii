#ifndef DEEP_LOOP_H
#define DEEP_LOOP_H

#include <vector>
#include <algorithm>
#include <iterator>

#include <deal.II/base/config.h>

DEAL_II_NAMESPACE_OPEN

template <int dim>
class DeepLoopCounter
{
public:
    //Constructors
	DeepLoopCounter()
	{
		std::fill_n(layer_counts, dim+1, 0);
	}
	DeepLoopCounter(const unsigned int sz) : sz(sz)
    {
		std::fill_n(layer_counts, dim+1, 0);
    }
	
	//Copy constructors and destructor
	DeepLoopCounter(const DeepLoopCounter& counter_old)
	{
		std::copy_n(counter_old.layer_counts, dim+1, layer_counts);
		sz = counter_old.sz;
	}
	DeepLoopCounter& operator=(const DeepLoopCounter& counter_old)
	{
		std::copy_n(counter_old.layer_counts, dim+1, layer_counts);
		sz = counter_old.sz;
		return *this;
	}
	~DeepLoopCounter() = default;
	
    //Check status of counter
<<<<<<< HEAD
    unsigned int layer_index(const unsigned int l) {return layer_counts[l];}
=======
    unsigned int layer_index(const unsigned int l) {return layer_counts[l];}    //Equivalent to subscript operator
>>>>>>> 26fca8321c53b404456112f7dc398fda51d28555
    unsigned int operator[](const unsigned int l) {return layer_counts[l];}
    
    bool is_active() {return layer_counts[dim] == 0;}
	bool is_finished() {return layer_counts[dim] != 0;}
	
	unsigned int get_global_index()
    {
        unsigned int index = 0;
        for (unsigned int d = dim; d > 0; --d)
        {
            index *= sz;
            index += layer_counts[d-1];
        }
        return index;
    }
    
    unsigned int size()
    {
        return sz;
    }
    
    //Comparison operators
    bool operator==(const DeepLoopCounter& counter2)
    {
        if (sz != counter2.sz) return false;
        for (unsigned int d = 0; d < dim+1; ++d)
            if (layer_counts[d] != counter2.layer_counts[d]) return false;
        return true;
    }
    
    bool operator<(const DeepLoopCounter& counter2)
    {
        if (sz < counter2.sz) return true;
        else if (sz > counter2.sz) return false;
        for (unsigned int d = dim+1; d > 0; --d)
        {
            if (layer_counts[d] < counter2.layer_counts[d]) return true;
            else if (layer_counts[d] > counter2.layer_counts[d]) return false;
        }
        return false;
    }
    bool operator<=(const DeepLoopCounter& counter2) {return ((*this)==counter2) || ((*this)<counter2);}
    
    bool operator>(const DeepLoopCounter& counter2) {return !((*this)<=counter2);}
    bool operator>=(const DeepLoopCounter& counter2) {return !((*this)<counter2);}
	
	//increment and decrement
	DeepLoopCounter& operator++()
    {
        for (unsigned int d = 0; d < dim+1; ++d)
        {
            if (++layer_counts[d] < sz) break;
            if (d == dim) break;
            layer_counts[d] = 0;
        }
        return *this;
    }
    DeepLoopCounter operator++(int)
    {
        DeepLoopCounter output(*this);
        ++(*this);
        return output;
    }
    
    DeepLoopCounter& operator--()
    {
        for (unsigned int d = 0; d < dim+1; ++d)
        {
            if (layer_counts[d] == 0) 
            {
                if (d == dim) break;
                layer_counts[d] = sz;
                continue;
            }
            --layer_counts[d];
            break;
        }
        return *this;
    }
    DeepLoopCounter operator--(int)
    {
        DeepLoopCounter output(*this);
        --(*this);
        return output;
    }
    
    //change loop 
    DeepLoopCounter& resize(unsigned int size)
    {
        sz = size;
        std::copy_n(std::vector<unsigned int>(dim+1).begin(), dim+1, layer_counts);
    }
    
    DeepLoopCounter& operator=(const unsigned int index_global)
    {
        unsigned int temp = index_global;
        for (unsigned int d = 0; (d < dim) && (temp != 0); ++d)
        {
            layer_counts[d] = temp % sz;
            temp -= layer_counts[d];
            temp /= sz;
        }
        
        if (temp != 0)
        {
            std::fill_n(layer_counts, dim, 0);
            layer_counts[dim] = 1;
        }
        
        return *this;
    }

private:
	unsigned int layer_counts[dim+1];
	unsigned int sz = 0;
};

DEAL_II_NAMESPACE_CLOSE

#endif
