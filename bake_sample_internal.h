#pragma once

#include <algorithm>
#include <cassert>

// Distributes samples per element (instance, triangle, ...), satisfying two constraints:
//  * total samples add up to num_samples
//  * minimum number of samples per element
// Any extra samples are placed according to area ratios.
// The template param is for specifying area per element, and min samples per element.
template <class T> 
void distribute_samples_generic(
  const T& callback,
  size_t num_samples,
  size_t num_elements,
  size_t* num_samples_per_element
  )
{
  // First place minimum samples per element
  size_t sample_count = 0;  // For all elements
  for (size_t i = 0; i < num_elements; ++i) {
    num_samples_per_element[i] = callback.minSamples(i);
    sample_count += num_samples_per_element[i];
  }
  
  assert(num_samples >= sample_count);

  if (num_samples > sample_count) {
    
    // Area-based sampling

    const size_t num_area_based_samples = num_samples - sample_count;

    // Compute surface area of each element
    double total_area = 0.0;
    for (size_t i = 0; i < num_elements; ++i) {
      total_area += callback.area(i);
    }
    
    // Distribute
    for (size_t i = 0; i < num_elements && sample_count < num_samples; ++i) {
      const size_t n = std::min(num_samples - sample_count, static_cast<size_t>(num_area_based_samples * callback.area(i) / total_area));
      num_samples_per_element[i] += n;
      sample_count += n;
    }

    // There could be a few samples left over. Place one sample per element until target sample count is reached.
    assert( num_samples - sample_count <= num_elements );
    for (size_t i = 0; i < num_elements && sample_count < num_samples; ++i) {
      num_samples_per_element[i] += 1;
      sample_count += 1;
    }
  }

  assert(sample_count == num_samples);

}

