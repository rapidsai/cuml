#include <gtest/gtest.h>
#include "linalg/add.h"
#include "linalg/subtract.h"
#include "cuda_utils.h"

#include <assert.h>
#include <math.h>
#include <stddef.h>

#include <vector>
#include <algorithm>
#include <limits>

namespace
{
    template<typename T>
    class TestBuffer
    {
    public:
        TestBuffer(size_t arrayLength)
        : devContainterRaw(nullptr)
        , hostContainter(arrayLength, T())
        {
            MLCommon::allocate(devContainterRaw, arrayLength);
            EXPECT_TRUE(devContainterRaw != nullptr);
        }

        ~TestBuffer() {
            EXPECT_TRUE(cudaFree(devContainterRaw) == cudaSuccess);
        }

        T* getDevPtr() {
            return devContainterRaw;
        }

        T* getHostPtr() {
            if (hostContainter.empty())
                return nullptr;
            else
                return &hostContainter[0];
        }

        T hostValueAt(size_t index) const {
            if (index >= hostContainter.size())
            {
                assert(!"INDEX IS OT OF ACCESSABLE RANGE");
                return T();
            }
            return hostContainter.at(index);
        }

        size_t size() const {
            return hostContainter.size();
        }

        void fillArithmeticSeq(const T& start = T(1), const T& step = T(1))
        {
            for (size_t i = 0; i < hostContainter.size(); ++i)
                hostContainter[i] = start + step*i;
            copy2Device();
        }

        void copy2Device() {
            EXPECT_TRUE(cudaMemcpy(getDevPtr(), getHostPtr(), size() * sizeof(T), cudaMemcpyHostToDevice) == cudaSuccess);
        }

        void copy2Host() {
            EXPECT_TRUE(cudaMemcpy(getHostPtr(), getDevPtr(), size() * sizeof(T), cudaMemcpyDeviceToHost) == cudaSuccess);
        }

    private:
        T* devContainterRaw;
        std::vector<T> hostContainter;

    private:
        TestBuffer(const TestBuffer&) = delete;
        TestBuffer operator = (const TestBuffer&) = delete;
    };
}

template<typename T>
void test_add(size_t arraLength)
{
    TestBuffer<T> in(arraLength);
    TestBuffer<T> extraScalar(1);
    TestBuffer<T> out(arraLength);
    in.fillArithmeticSeq();
    extraScalar.fillArithmeticSeq();
    out.fillArithmeticSeq();

    MLCommon::LinAlg::addDevScalar(out.getDevPtr(), in.getDevPtr(), extraScalar.getDevPtr(), in.size());
    out.copy2Host();

    T maxError = T();
    for (int i = 0; i < arraLength; i++)
    {
        maxError = std::max(maxError,
                            abs( (in.hostValueAt(i) + extraScalar.hostValueAt(0)) - out.hostValueAt(i) )
                           );
    }
    EXPECT_TRUE(maxError < std::numeric_limits<T>::epsilon()) << "Max deviation in test_add is greater then " << std::numeric_limits<T>::epsilon();
}

template<typename T>
void test_subtract(size_t arraLength)
{
    TestBuffer<T> in(arraLength);
    TestBuffer<T> extraScalar(1);
    TestBuffer<T> out(arraLength);
    in.fillArithmeticSeq();
    extraScalar.fillArithmeticSeq();
    out.fillArithmeticSeq();

    MLCommon::LinAlg::subtractDevScalar(out.getDevPtr(), in.getDevPtr(), extraScalar.getDevPtr(), in.size());
    out.copy2Host();

    T maxError = T();
    for (int i = 0; i < arraLength; i++)
        maxError = std::max(maxError,
                            abs( (in.hostValueAt(i) - extraScalar.hostValueAt(0)) - out.hostValueAt(i) )
                           );
    EXPECT_TRUE(maxError < std::numeric_limits<T>::epsilon()) << "Max deviation test_subtract is greater then " << std::numeric_limits<T>::epsilon();
}

TEST(AddAndSubDevScalarTest, add_test)
{
    test_add<float>(1);
    test_add<float>(100);
    test_add<double>(1);
    test_add<double>(100);
}

TEST(AddAndSubDevScalarTest, subtract_test)
{
    test_subtract<float>(1);
    test_subtract<float>(100);
    test_subtract<double>(1);
    test_subtract<double>(100);
}
