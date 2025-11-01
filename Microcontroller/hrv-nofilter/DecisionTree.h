#pragma once
namespace Eloquent {
    namespace ML {
        namespace Port {
            class DecisionTree {
                public:
                    /**
                    * Predict class for features vector
                    */
                    int predict(float *x) {
                        if (x[1] < 50.0) {
                            return 1;
                        } else if (x[1] >= 50.0) {
                            if (x[0] <= 749.34) {
                                return 1;
                            } else {
                                return 0;
                            }
                        }
                    }
                protected:
            };
        }
    }
}