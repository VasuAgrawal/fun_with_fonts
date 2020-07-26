#pragma once

#include <string_view>
#include <algorithm>

template<class I>
auto prefix_length
(
	I abegin, I aend,
	I bbegin, I bend
)
{
    return std::distance(abegin, std::mismatch(abegin, aend, bbegin, bend).first);
}

static inline
auto reduce
(
	std::string_view& a,
	std::string_view& b
)
	-> void
{
	auto prefix = prefix_length(a.begin(), a.end(), b.begin(), b.end());
	a.remove_prefix(prefix);
	b.remove_prefix(prefix);

    auto suffix = prefix_length(a.rbegin(), a.rend(), b.rbegin(), b.rend());
	a.remove_suffix(suffix);
    b.remove_suffix(suffix);
}
