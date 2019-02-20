#pragma once

namespace ML {
    class UMAPParams {
    public:

        int n_neighbors = 15;
        int n_components = 2;
        int n_epochs = 5;

        float learning_rate = 1.0;
        float min_dist = 0.1;
        float spread = 1.0;
        float set_op_mix_ratio = 1.0;

        int local_connectivity = 1;
        float repulsion_strength = 1.0;
        int negative_sample_rate = 5;
        float transform_queue_size = 4.0;


        /**
         * Parameters of differentiable approx
         * of right adjoint functor.
         */
        float a, b;

        float gamma;

        int target_n_neighbors = -1;
        float target_weight = 0.5;

        /**
         * Initial learning rate for SGD
         */
        float initial_alpha = 1.0;
    };

}
