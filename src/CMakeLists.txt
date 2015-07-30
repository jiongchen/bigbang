file(GLOB header *.h)
file(GLOB source *.cc)

add_library(bigbang SHARED ${header} ${source})
