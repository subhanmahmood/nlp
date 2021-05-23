import requests
import json

def test_valid_body_single():
    url = 'http://localhost:5000/predict'

    headers = {'Content-Type': 'application/json'}

    payload = [    
        {
            "title": "Night of the Wolf: Late Phases",
            "plot": "When deadly beasts attack from the forest, it is up to a grizzled veteran to uncover what the residents of a secluded retirement community are hiding."
        }
    ]

    res = requests.post(url, headers=headers, data=json.dumps(payload))

    assert res.status_code == 200
    res_body = res.json()

    assert len(res_body) == 1
    assert res.url == url

def test_valid_body_mutliple():
    url = 'http://localhost:5000/predict'

    headers = {'Content-Type': 'application/json'}

    payload = [    
        {
            "title": "Night of the Wolf: Late Phases",
            "plot": "When deadly beasts attack from the forest, it is up to a grizzled veteran to uncover what the residents of a secluded retirement community are hiding."
        },
        {
            "title": "Stake Land",
            "plot": "In a world of vampires, an expert vampire hunter and his young protégé travel toward sanctuary."
        }
    ]

    res = requests.post(url, headers=headers, data=json.dumps(payload))

    assert res.status_code == 200
    res_body = res.json()

    assert len(res_body) == 2
    assert res.url == url

def test_invalid_body_missing_title():

    url = 'http://localhost:5000/predict'

    headers = {'Content-Type': 'application/json'}

    payload = [
        {
            "plot": "When deadly beasts attack from the forest, it is up to a grizzled veteran to uncover what the residents of a secluded retirement community are hiding."
        }
    ]

    res = requests.post(url, headers=headers, data=json.dumps(payload))

    assert res.status_code == 406
    res_body = res.json()

    assert res.url == url

def test_invalid_body_missing_plot():

    url = 'http://localhost:5000/predict'

    headers = {'Content-Type': 'application/json'}

    payload = [
        {
            "title": "Night of the Wolf: Late Phases",
        }
    ]

    res = requests.post(url, headers=headers, data=json.dumps(payload))

    assert res.status_code == 406
    res_body = res.json()

    assert res.url == url


def test_invalid_body_invalid_title():

    url = 'http://localhost:5000/predict'

    headers = {'Content-Type': 'application/json'}

    payload = [
        {
            "title": 1,
            "plot": "When deadly beasts attack from the forest, it is up to a grizzled veteran to uncover what the residents of a secluded retirement community are hiding."
        }
    ]

    res = requests.post(url, headers=headers, data=json.dumps(payload))

    assert res.status_code == 406
    res_body = res.json()

    assert res.url == url

def test_invalid_body_invalid_title():

    url = 'http://localhost:5000/predict'

    headers = {'Content-Type': 'application/json'}

    payload = [
        {
            "title": "Night of the Wolf: Late Phases",
            "plot": 1
        }
    ]

    res = requests.post(url, headers=headers, data=json.dumps(payload))

    assert res.status_code == 406
    res_body = res.json()

    assert res.url == url

def test_invalid_body_empty():
    url = 'http://localhost:5000/predict'

    headers = {'Content-Type': 'application/json'}

    payload = []

    res = requests.post(url, headers=headers, data=json.dumps(payload))

    assert res.status_code == 406
    res_body = res.json()

    assert res.url == url

def test_invalid_body():
    url = 'http://localhost:5000/predict'

    headers = {'Content-Type': 'application/json'}

    payload = 'test'

    res = requests.post(url, headers=headers, data=json.dumps(payload))

    assert res.status_code == 406
    res_body = res.json()

    assert res.url == url
.pytest_cache/**
_pycache__