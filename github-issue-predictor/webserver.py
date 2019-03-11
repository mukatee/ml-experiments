__author__ = 'teemu kanstren'

from bottle import run, request, post, get
from arm.rf import predict_component

#for making predictions for given issue_id. load issue text from github, try to predict the component probabilities for it
@post('/')
def index():
	postdata = request.body.read()
	print(postdata)  # this goes to log file only, not to client
	issue = request.forms.get("issue_id")
	probabilities = predict_component(issue)
	form = '<form action="/" method="post">Issue id:<input type="text" name="issue_id" value=""><input type="submit" value="Arvaus"></form>'
	page = form
	for probability in probabilities:
		page += "<br>"+probability[0]+": "+str(probability[1])
	return page

#for the first page load without form submission
@get('/')
def index():
	form = '<form action="/" method="post">Issue id:<input type="text" name="issue_id" value=""><input type="submit" value="Arvaus"></form>'
	page = form
	return page

run(host='localhost', port=8080, debug=True)
