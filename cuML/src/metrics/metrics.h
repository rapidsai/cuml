/*
 * metrics.h
 *
 *  Created on: Apr 2, 2019
 *      Author: cjnolet
 */

#pragma once

#include "metrics/metrics.h"

namespace ML {

    namespace Metrics {

        /**
         * Calculates the "Coefficient of Determination" (R-Squared) score
         * normalizing the sum of squared errors by the total sum of squares.
         *
         * This score indicates the proportionate amount of variation in an
         * expected response variable is explained by the independent variables
         * in a linear regression model. The larger the R-squared value, the
         * more variability is explained by the linear regression model.
         *
         * @param y: Array of ground-truth response variables
         * @param y_hat: Array of predicted response variables
         * @param n: Number of elements in y and y_hat
         * @return: The R-squared value.
         */
        template <typename T>
        T r_squared(T *y, T *y_hat, int n) {
            return MLCommon::Metrics::r_squared(y, y_hat, n);
        }
    }

}
