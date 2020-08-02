from django.conf.urls import patterns,url
from RLCchess import views

urlpatterns = patterns('',
						url(r'^$',views.index,name='index'),
						url(r'^play_game/$',views.play_game,name='play_game'),)