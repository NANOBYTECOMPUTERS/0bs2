#pragma once

#include <cmath>
#include <exception>
#include <iostream>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

namespace obs2test
{
class TestFailure : public std::exception
{
public:
    explicit TestFailure(std::string message)
        : message_(std::move(message))
    {
    }

    const char* what() const noexcept override
    {
        return message_.c_str();
    }

private:
    std::string message_;
};

inline void require(bool condition, const char* expression, const char* file, int line)
{
    if (condition)
        return;

    std::ostringstream oss;
    oss << file << ":" << line << ": requirement failed: " << expression;
    throw TestFailure(oss.str());
}

inline void requireNear(double actual, double expected, double tolerance, const char* expression, const char* file, int line)
{
    if (std::isfinite(actual) && std::fabs(actual - expected) <= tolerance)
        return;

    std::ostringstream oss;
    oss << file << ":" << line << ": " << expression << " expected " << expected
        << " +/- " << tolerance << ", got " << actual;
    throw TestFailure(oss.str());
}

struct NamedTest
{
    const char* name;
    void (*run)();
};

inline int runTests(const std::vector<NamedTest>& tests, const char* suiteName)
{
    int failures = 0;
    for (const auto& test : tests)
    {
        try
        {
            test.run();
            std::cout << "[PASS] " << test.name << "\n";
        }
        catch (const std::exception& e)
        {
            ++failures;
            std::cerr << "[FAIL] " << test.name << ": " << e.what() << "\n";
        }
    }

    if (failures != 0)
    {
        std::cerr << failures << " " << suiteName << " test(s) failed.\n";
        return 1;
    }

    std::cout << tests.size() << " " << suiteName << " tests passed.\n";
    return 0;
}
}

#define REQUIRE(expr) obs2test::require((expr), #expr, __FILE__, __LINE__)
#define REQUIRE_NEAR(actual, expected, tolerance) obs2test::requireNear((actual), (expected), (tolerance), #actual, __FILE__, __LINE__)
