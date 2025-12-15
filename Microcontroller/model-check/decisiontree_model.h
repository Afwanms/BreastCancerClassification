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
                        if (x[0] <= 1.9009275436401367) {
                            if (x[0] <= 0.42482107877731323) {
                                if (x[1] <= -0.26795494556427) {
                                    return 1;
                                }

                                else {
                                    return 0;
                                }
                            }

                            else {
                                if (x[1] <= 0.5605651140213013) {
                                    return 1;
                                }

                                else {
                                    return 0;
                                }
                            }
                        }

                        else {
                            return 0;
                        }
                    }

                protected:
                };
            }
        }
    }