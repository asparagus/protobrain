# #!/usr/bin/python
# # -*- coding: utf-8 -*-
# import pytest
# import numpy as np
# from protobrain import event
# from protobrain import sensor


# def test_wrong_length():
#     s = sensor.Sensor(10)
#     with pytest.raises(ValueError):
#         s.set_values(np.zeros(3))

# def test_emit():
#     s = sensor.Sensor(10)
#     verify = event.EventVerifier(s.emit)

#     s.set_values(np.zeros(10))

#     assert verify.has_run
