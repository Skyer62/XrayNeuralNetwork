from main import predict

def test_pred_Normal():
    assert predict() != "Норма"

def test_pred_Pneumonia():
    assert predict() != "Пневмония"

def test_pred_Tuberculesus():
    assert predict() != "Туберкулёз"

def test_pred_Covid_19():
    assert predict() == "COVID-19"