import test_dansy_analysis as test1
import test_dedansy_analysis as test2

def run_tests():
    print('Testing the base DANSy analysis')
    test1.main()

    print('Testing the deDANSy analysis')
    test2.main()

if __name__ == '__main__':
    run_tests()
    
    