from django.shortcuts import render,render_to_response
from django.http import HttpResponse
from django.template import RequestContext

def index(request):
	context = RequestContext(request)

	return render_to_response('RLCchess/index.html',{},context)
