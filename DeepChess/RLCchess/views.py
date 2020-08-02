from django.shortcuts import render,render_to_response
from django.http import HttpResponse
from django.template import RequestContext

def index(request):
	context = RequestContext(request)

	context_dict={}
	context_dict['move'] = "e7-e5"

	return render_to_response('RLCchess/index.html',context_dict,context)
