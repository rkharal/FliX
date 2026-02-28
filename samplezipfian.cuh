#include <boost/random/zipf_distribution.hpp>
//------------------------------------

These are working zip functions


template <typename key_type>
void generate_keys_zipf_file(
    size_t size,
    key_type min_usable_key,
    key_type max_usable_key,
    std::vector<key_type>& generated_keys,
    const std::string& filename = "zipf_keys_cache.txt"
) {
    // Try to load from cache
    std::ifstream infile(filename);
    if (infile.good()) {
        key_type key;
        while (infile >> key) {
            generated_keys.push_back(key);
            if (generated_keys.size() >= size) break;
        }
        infile.close();
        if (generated_keys.size() == size) {
            std::cerr << "Loaded keys from cache: " << filename << std::endl;
            return;
        }
    }

    // If no valid cache, generate new keys
    std::cerr << "Generating new Zipfian keys..." << std::endl;
    std::mt19937_64 gen(42); // Ensuring deterministic behavior
    size_t range = max_usable_key - min_usable_key + 1;
    
    // Precompute Zipfian probabilities
    std::vector<double> probabilities(range);
    double zipfian_s = 2.0; // Zipfian skewness parameter
    double sum = 0.0;
    for (size_t i = 1; i <= range; ++i) {
        sum += 1.0 / std::pow(i, zipfian_s);
    }
    for (size_t i = 1; i <= range; ++i) {
        probabilities[i - 1] = (1.0 / std::pow(i, zipfian_s)) / sum;
    }
    
    std::discrete_distribution<size_t> zipf_dist(probabilities.begin(), probabilities.end());
    std::unordered_set<key_type> generated_keys_set;

    while (generated_keys_set.size() < size) {
        key_type key = min_usable_key + zipf_dist(gen);
        generated_keys_set.insert(key);
    }

    // Convert set to vector
    generated_keys.assign(generated_keys_set.begin(), generated_keys_set.end());

    // Save to cache file
    std::ofstream outfile(filename);
    for (const auto& key : generated_keys) {
        outfile << key << "\n";
    }
    outfile.close();

    std::cerr << "Saved generated keys to cache: " << filename << std::endl;
}

template <typename key_type>
void generate_keys_zipf_prev(
    size_t size,
    key_type min_usable_key,
    key_type max_usable_key,
    std::vector<key_type>& generated_keys
) {
    std::mt19937_64 gen(42); // Ensuring deterministic behavior
    size_t range = max_usable_key - min_usable_key + 1;
    
    // Generate Zipfian probabilities
    std::vector<double> probabilities(range);
    double zipfian_s = 2.0; // Zipfian skewness parameter
    double sum = 0.0;
    for (size_t i = 1; i <= range; ++i) {
        sum += 1.0 / std::pow(i, zipfian_s);
    }
    for (size_t i = 1; i <= range; ++i) {
        probabilities[i - 1] = (1.0 / std::pow(i, zipfian_s)) / sum;
    }
    
    std::discrete_distribution<size_t> zipf_dist(probabilities.begin(), probabilities.end());
    std::unordered_set<key_type> generated_keys_set; // Ensuring unique keys
    
    while (generated_keys_set.size() < size) {
        key_type key = min_usable_key + zipf_dist(gen);
        generated_keys_set.insert(key);
    }
    
    generated_keys.assign(generated_keys_set.begin(), generated_keys_set.end());
}




//-----------------------------------



template <typename key_type>
void generate_keys_zipfian(
    size_t size,
    key_type min_usable_key,
    key_type max_usable_key,
    std::vector<key_type>& generated_keys,
    double zipf_alpha = 1.2 // Zipfian skewness parameter
) {
    std::mt19937_64 gen(42); // Fixed seed for reproducibility
    boost::random::zipf_distribution<key_type> zipf_dist(max_usable_key - min_usable_key + 1, zipf_alpha);

    std::unordered_set<key_type> generated_keys_set;
    while (generated_keys_set.size() < size) {
        key_type key = min_usable_key + zipf_dist(gen);
        generated_keys_set.insert(key);
    }

    generated_keys.resize(size);
    std::copy(generated_keys_set.begin(), generated_keys_set.end(), generated_keys.begin());
}



#include <random>
#include <unordered_set>
#include <vector>
#include <cmath>

// Zipfian distribution sampling function
template <typename key_type>
key_type zipfian_sample(key_type min_key, key_type max_key, double alpha, std::mt19937_64& gen) {
    static std::vector<double> probabilities;
    static bool initialized = false;
    static key_type range_size = max_key - min_key + 1;

    if (!initialized) {
        // Precompute cumulative distribution
        probabilities.resize(range_size + 1);
        double sum = 0.0;
        for (key_type i = 1; i <= range_size; ++i) {
            sum += 1.0 / std::pow(i, alpha);
            probabilities[i] = sum;
        }
        for (key_type i = 1; i <= range_size; ++i) {
            probabilities[i] /= sum; // Normalize
        }
        initialized = true;
    }

    // Generate a random number in [0,1)
    std::uniform_real_distribution<double> dist(0.0, 1.0);
    double random_value = dist(gen);

    // Find the corresponding Zipfian-distributed index using binary search
    auto it = std::lower_bound(probabilities.begin(), probabilities.end(), random_value);
    key_type index = std::distance(probabilities.begin(), it);
    
    return min_key + index - 1;
}

// Function to generate keys using Zipfian distribution
template <typename key_type>
void generate_keys_zipf(
    size_t size,
    key_type min_usable_key,
    key_type max_usable_key,
    double alpha, // Zipfian distribution skew factor
    std::vector<key_type>& generated_keys
) {
    std::mt19937_64 gen(42); // Fixed seed for reproducibility
    std::unordered_set<key_type> generated_keys_set;

    while (generated_keys_set.size() < size) {
        key_type new_key = zipfian_sample(min_usable_key, max_usable_key, alpha, gen);
        generated_keys_set.insert(new_key);
    }

    // Copy unique generated keys to vector
    generated_keys.resize(size);
    std::copy(generated_keys_set.begin(), generated_keys_set.end(), generated_keys.begin());
}


std::vector<int> keys;
generate_keys_zipf(1000, 1, 100000, 1.2, keys);




-------#include <fstream>
#include <iostream>
#include <vector>
#include <random>
#include <unordered_set>
#include <cmath>

template <typename key_type>
void generate_keys_zipf(
    size_t size,
    key_type min_usable_key,
    key_type max_usable_key,
    std::vector<key_type>& generated_keys,
    const std::string& filename = "zipf_keys_cache.txt"
) {
    // Try to load from cache
    std::ifstream infile(filename);
    if (infile.good()) {
        key_type key;
        while (infile >> key) {
            generated_keys.push_back(key);
            if (generated_keys.size() >= size) break;
        }
        infile.close();
        if (generated_keys.size() == size) {
            std::cerr << "Loaded keys from cache: " << filename << std::endl;
            return;
        }
    }

    // If no valid cache, generate new keys
    std::cerr << "Generating new Zipfian keys..." << std::endl;
    std::mt19937_64 gen(42); // Ensuring deterministic behavior
    size_t range = max_usable_key - min_usable_key + 1;
    
    // Precompute Zipfian probabilities
    std::vector<double> probabilities(range);
    double zipfian_s = 2.0; // Zipfian skewness parameter
    double sum = 0.0;
    for (size_t i = 1; i <= range; ++i) {
        sum += 1.0 / std::pow(i, zipfian_s);
    }
    for (size_t i = 1; i <= range; ++i) {
        probabilities[i - 1] = (1.0 / std::pow(i, zipfian_s)) / sum;
    }
    
    std::discrete_distribution<size_t> zipf_dist(probabilities.begin(), probabilities.end());
    std::unordered_set<key_type> generated_keys_set;

    while (generated_keys_set.size() < size) {
        key_type key = min_usable_key + zipf_dist(gen);
        generated_keys_set.insert(key);
    }

    // Convert set to vector
    generated_keys.assign(generated_keys_set.begin(), generated_keys_set.end());

    // Save to cache file
    std::ofstream outfile(filename);
    for (const auto& key : generated_keys) {
        outfile << key << "\n";
    }
    outfile.close();

    std::cerr << "Saved generated keys to cache: " << filename << std::endl;
}



Yes! There are two optimizations you can make:

1. **Avoid recomputing Zipfian probabilities repeatedly** – Compute them once and save them.
2. **Cache and load the generated keys from a file** instead of regenerating them every time.

---

### **Optimized `generate_keys_zipf` with File Caching**
This version:
- **Generates keys only if they don’t exist** in a file.
- **Saves the generated keys** to a file (`zipf_keys_cache.txt`).
- **Loads them quickly from disk** the next time.

#### **Updated Code:**
```cpp
#include <fstream>
#include <iostream>
#include <vector>
#include <random>
#include <unordered_set>
#include <cmath>

template <typename key_type>
void generate_keys_zipf(
    size_t size,
    key_type min_usable_key,
    key_type max_usable_key,
    std::vector<key_type>& generated_keys,
    const std::string& filename = "zipf_keys_cache.txt"
) {
    // Try to load from cache
    std::ifstream infile(filename);
    if (infile.good()) {
        key_type key;
        while (infile >> key) {
            generated_keys.push_back(key);
            if (generated_keys.size() >= size) break;
        }
        infile.close();
        if (generated_keys.size() == size) {
            std::cerr << "Loaded keys from cache: " << filename << std::endl;
            return;
        }
    }

    // If no valid cache, generate new keys
    std::cerr << "Generating new Zipfian keys..." << std::endl;
    std::mt19937_64 gen(42); // Ensuring deterministic behavior
    size_t range = max_usable_key - min_usable_key + 1;
    
    // Precompute Zipfian probabilities
    std::vector<double> probabilities(range);
    double zipfian_s = 2.0; // Zipfian skewness parameter
    double sum = 0.0;
    for (size_t i = 1; i <= range; ++i) {
        sum += 1.0 / std::pow(i, zipfian_s);
    }
    for (size_t i = 1; i <= range; ++i) {
        probabilities[i - 1] = (1.0 / std::pow(i, zipfian_s)) / sum;
    }
    
    std::discrete_distribution<size_t> zipf_dist(probabilities.begin(), probabilities.end());
    std::unordered_set<key_type> generated_keys_set;

    while (generated_keys_set.size() < size) {
        key_type key = min_usable_key + zipf_dist(gen);
        generated_keys_set.insert(key);
    }

    // Convert set to vector
    generated_keys.assign(generated_keys_set.begin(), generated_keys_set.end());

    // Save to cache file
    std::ofstream outfile(filename);
    for (const auto& key : generated_keys) {
        outfile << key << "\n";
    }
    outfile.close();

    std::cerr << "Saved generated keys to cache: " << filename << std::endl;
}
```

---

### **How It Works**
1. **First Run:**
   - Tries to **load keys from a file** (`zipf_keys_cache.txt`).
   - If the file **exists and contains the correct number of keys**, it **loads them instantly**.
   - If not, it **generates new Zipfian-distributed keys** and **saves them** to the file.

2. **Subsequent Runs:**
   - Simply **reads the pre-generated keys from disk**, which is **MUCH faster**.

---

### **Performance Boost**
- **File caching** eliminates the need for recomputation.
- **Precomputed probabilities** improve efficiency.
- **No sorting or unnecessary operations** on the generated set.

---

### **Usage Example**
```cpp
std::vector<int> keys;
generate_keys_zipf(100000, 1, 1000000, keys); // Fast on subsequent runs!
```

Would you like an option to **invalidate the cache** if needed? 🚀