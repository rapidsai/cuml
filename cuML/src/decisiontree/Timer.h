#pragma once

#include <chrono>


class Timer
{
public:
    Timer()
    {
        this->reset();
    }

    void reset()
    {
        this->time  = std::chrono::high_resolution_clock::now();
    }

        double getElapsedSeconds() const
    {
        return 1.0e-6 * std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - this->time).count();
    }

    double getElapsedMilliseconds() const
    {
        return 1.0e-3 * std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - this->time).count();
    }

private:
    std::chrono::high_resolution_clock::time_point time;
};
