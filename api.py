#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 23 09:34:59 2020

@author: santiago
"""

from flask import Flask, request
# import pudb
import json
app = Flask(__name__)

@app.route('/', methods=['GET'])
def index():      
      return {"franquito": "vaaaamo bokita!", 
              "luna": "me estresé", 
              "rebo": "Greetings sato."}

@app.route('/results', methods=['POST'])
def results():
      data = json.loads(request.get_data())
      params = data.get("params")
      # PROCESAR LA MIERDA ESTA 
      
      return {"franquito": "vaaaamo bokita!", 
              "luna": "me estresé", 
              "rebo": "Greetings sato."}

if __name__ == "__main__":
        app.run(debug = True, port = 8080)

