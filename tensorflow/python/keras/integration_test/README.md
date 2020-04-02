# Keras Integration Test

This package contains integration tests that ensure the correct interaction
between Keras and other Tensorflow high level APIs, like dataset, TF function
and distribution strategy, etc.

There are a few guidelines for the tests under this package.

_. Only use the public TF API. _. Test should focus on the end to end use case
between Keras and other TF high level API. Unit test will be a better place for
behavior testing for the individual APIs.
