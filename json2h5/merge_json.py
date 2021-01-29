with open ('/media/hkuit155/8221f964-4062-4f55-a2f3-78a6632f7418/Autoannotation_Pose/json2h5/json/ai_add_searchedyoga_test.json') as f1:
    f1data = f1.read()
with open('/media/hkuit155/8221f964-4062-4f55-a2f3-78a6632f7418/Autoannotation_Pose/json2h5/json/ai_add_searchedyoga_train.json') as f2:
    f2data = f2.read()
f1data += "/n"
f1data += f2data
with open('/media/hkuit155/8221f964-4062-4f55-a2f3-78a6632f7418/Autoannotation_Pose/json2h5/json/ai_add_searchedyoga.json','a') as f3:
    f3.write(f1data)