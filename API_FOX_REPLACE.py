

#==============================================================================
# Author : Kent Chiu (kentchun33333@gmail.com)
#==============================================================================
# 
# (API_FOX_REPLACE.py) 
# - Used for replace the doc_key_words for hidden Informattion in Foxcon
# - cd to the doc_folder to replace it
# - KentChiu_DocFactory is Powered by epydoc
# 
#==============================================================================

import os
import sys
import argparse


replacements = {'epydoc':'KentChiu_DocFactory',
    'Epydoc':'KentChiu_DocFactory',
    'source&nbsp;code':'All Right Reserved',
    'sourceforge.net':'',
    'http://':''}

files = os.listdir(os.getcwd())

for file in files :
	if file.split('.')[-1] in ('html','js','css'):
		lines = []
		with open(file) as f:
			for line in f:
				for src, target in replacements.iteritems():
					line = line.replace(src, target)
				lines.append(line)

		with open(file, 'w') as f:
			for line in lines:
				f.write(line)

	if file.split('.')[-1] in ('js', 'css'):
		for src, target in replacements.iteritems():
			newName = file.replace(src, target)
			os.rename(file, newName)
