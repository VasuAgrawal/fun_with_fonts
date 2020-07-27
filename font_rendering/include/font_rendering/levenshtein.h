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

class Utf8Iterator {
  // Borrowing from this very helpful answer:
  // https://stackoverflow.com/a/8054856/893643
 public:
  using iterator_category = std::forward_iterator_tag;
  using value_type = std::string;
  using difference_type = std::ptrdiff_t;
  using pointer = const std::string*;
  using reference = const std::string&;

  Utf8Iterator(std::string_view::const_iterator it) : it_(it) {}

  Utf8Iterator& operator++() {  // prefix increment
    storeCurrent();
    it_ += c_.size();
    return *this;
  }

  reference operator*() const {
    storeCurrent();
    return c_;
  }

  bool operator==(const Utf8Iterator& other) const { return it_ == other.it_; }

  bool operator!=(const Utf8Iterator& other) const { return !(*this == other); }

 private:
  std::string_view::const_iterator it_;
  mutable std::string c_;

  void storeCurrent() const {
    // First, figure out how many bytes the current character is.
    const auto c = *it_;
    if ((c & 0b10000000) == 0) {  // Ascii
      c_ = value_type(it_, it_ + 1);
    } else if ((c & 0b11100000) == 0b11000000) {  // 2 byte character
      c_ = value_type(it_, it_ + 2);
    } else if ((c & 0b11110000) == 0b11100000) {  // 3 byte character
      c_ = value_type(it_, it_ + 3);
    } else if ((c & 0b11111000) == 0b11110000) {  // 4 byte character
      c_ = value_type(it_, it_ + 4);
    }
  }
};

class Utf8Adapter {
 public:
  using const_iterator = Utf8Iterator;
  // Not marked as explicit
  Utf8Adapter(std::string_view view = "") : view_(view) {}

  const_iterator begin() const { return const_iterator(view_.begin()); }

  const_iterator cbegin() const { return begin(); }

  const_iterator end() const { return const_iterator(view_.end()); }

  const_iterator cend() const { return end(); }

 private:
  std::string_view view_;
};

// Return a new string with lowercase ascii characters. Supports UTF8 encoding.
std::string lowerAsciiInUtf8(std::string_view s) {
  std::string output;
  output.reserve(s.size());
  for (const auto& c : Utf8Adapter(s)) {
    if (c.size() == 1) {
      if ('A' <= c[0] && c[0] <= 'Z') {
        output += c[0] - 'A' + 'a';
      } else {
        output += c[0];
      }
    } else {
      output += c;
    }
  }

  return output;
}
