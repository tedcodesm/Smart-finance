import mongomock, datetime
c = mongomock.MongoClient().db
# indexes
c1 = c["alerts"]
try:
    print("sparse unique:", c1.create_index([("user_id",1),("dedupe_key",1)], unique=True, sparse=True))
except Exception as e:
    print("sparse unique FAILED:", e)
try:
    c1.insert_one({"user_id":1,"dedupe_key":"a"})
    c1.insert_one({"user_id":1})
    c1.insert_one({"user_id":1})
    print("sparse allows missing-key dups: OK", c1.count_documents({}))
except Exception as e:
    print("sparse dup-missing FAILED:", e)
try:
    c2 = c["categories"]
    c2.create_index([("name",1)], unique=True, partialFilterExpression={"is_system":True})
    c2.insert_one({"name":"A","is_system":True})
    c2.insert_one({"name":"A","is_system":False})
    c2.insert_one({"name":"A","is_system":False})
    print("partial unique OK", c2.count_documents({}))
except Exception as e:
    print("partial unique FAILED:", e)
# aggregation stages
docs=[]
for i,(k,t,a) in enumerate([("2026-01","expense",10.0),("2026-01","income",100.0),("2026-02","expense",20.0)]):
    docs.append({"user_id":"u1","period_key":k,"type":t,"amount":a,"category_id":f"c{i%2}","date":datetime.datetime(2026,1,1)})
c3=c["tx"]; c3.insert_many(docs)
try:
    r=list(c3.aggregate([{"$match":{"user_id":"u1"}},{"$group":{"_id":{"pk":"$period_key","t":"$type"},"total":{"$sum":"$amount"},"n":{"$sum":1}}},{"$sort":{"_id.pk":1}}]))
    print("group nested OK:", r)
except Exception as e:
    print("group nested FAILED:", e)
try:
    r=list(c3.aggregate([{"$match":{"user_id":"u1"}},{"$group":{"_id":"$period_key","total":{"$sum":"$amount"}}},{"$sort":{"_id":1}},{"$project":{"_id":0,"period_key":"$_id","total":1}}]))
    print("project OK:", r)
except Exception as e:
    print("project FAILED:", e)
try:
    r=list(c3.aggregate([{"$match":{"user_id":"u1"}},{"$group":{"_id":None,"avg":{"$avg":"$amount"},"mx":{"$max":"$amount"}}}]))
    print("avg/max OK:", r)
except Exception as e:
    print("avg/max FAILED:", e)
try:
    r=c3.find_one_and_update({"user_id":"u1"},{"$set":{"x":1}},return_document=True)
    print("foau OK:", bool(r))
except Exception as e:
    print("foau FAILED:", e)
try:
    r=c3.bulk_write([__import__("pymongo").InsertOne({"user_id":"u2","amount":1})])
    print("bulk OK")
except Exception as e:
    print("bulk FAILED:", e)
