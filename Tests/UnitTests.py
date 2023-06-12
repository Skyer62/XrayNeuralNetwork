import unittest
from main import predict

class MyTestCase(unittest.TestCase):
    def test_something(self):
        self.assertEqual(predict(), "COVID-19")

if __name__ == '__main__':
    unittest.main()
