c = console.log.bind console


{ spawn, exec } = require 'child_process'


#vert = exec ('glslc.exe ./shaders/s1.vert -o ./spv/s1.vert.spv'), (err) ->
 #       c 'hello?', err


#vert = exec ('glslc.exe ./shaders/s_200__.vert -o ./spv/s_200__.vert.spv'), (err) ->
#        c 'hello?', err

vert = exec ('glslc.exe ./shaders/s_300_.vert -o ./spv/s_300__.vert.spv'), (err) ->
        if err then c 'errors:', err

vert = exec ('glslc.exe ./shaders/s_400_.vert -o ./spv/s_400_.vert.spv'), (err) ->
        if err then c 'errors:', err

frag = exec ('glslc.exe ./shaders/s1.frag -o ./spv/s1.frag.spv')



# vert.stderr.on 'data', (data) ->
#     c data
# glslc.exe ./shaders/src/shader.vert -o ./shaders/spv/vert.spv
# glslc.exe ./shaders/src/shader.frag -o ./shaders/spv/frag.spv
# pause
