#include <iostream>
#include <string_view>
#include <vector>

static constexpr size_t index(size_t row, size_t col, size_t width) {
  return row * width + col;
}

template <typename T>
static void printMatrix(const std::vector<T>& m, size_t width, size_t) {
  for (int i = 0; i < m.size(); ++i) {
    if (i && (i % width) == 0) {
      std::cout << "\n";
    }

    std::cout << m[i] << " ";
  }

  std::cout << std::endl;
  std::cout << std::endl;
}

size_t levenshteinNaive(std::string_view source, std::string_view target) {
  // Ripped from
  // https://people.cs.pitt.edu/~kirk/cs1501/Pruhs/Spring2006/assignments/editdistance/Levenshtein%20Distance.htm
  const size_t n = source.size();
  const size_t m = target.size();

  if (n == 0 || m == 0) {
    return 0;
  }

  const size_t width = n + 1;
  const size_t height = m + 1;
  std::vector<size_t> d(width * height);
  // Set first row to 0 to n
  for (int i = 0; i <= n; ++i) {
    d[index(0, i, width)] = i;
  }

  // Set the first column to 0 to m
  for (int i = 0; i <= m; ++i) {
    d[index(i, 0, width)] = i;
  }

  for (size_t i = 1; i <= source.size(); ++i) {
    for (size_t j = 1; j <= target.size(); ++j) {
      const size_t cost = source[i - 1] != target[j - 1];
      d[index(j, i, width)] = std::min(std::min(d[index(j, i - 1, width)] + 1,
                                                d[index(j - 1, i, width)] + 1),
                                       d[index(j - 1, i - 1, width)] + cost);
    }
  }

  return d[index(m, n, width)];
}
