import sys
import unittest
from unittest.mock import MagicMock, patch

sys.modules['cv2'] = MagicMock()
import foo

class TestFoo(unittest.TestCase):

     @patch('cv2.imread')
     def test_load_image(self, imread):
         foo.load_image('test')
         assert imread.called

if __name__ == '__main__':
    unittest.main()
