CXX = g++
project = main
#LDFLAGS = -L$(VULKAN_SDK_PATH)/lib `pkg-config --static --libs glfw3` -lvulkan
LDFLAGS = -lglfw -lxcb -lvulkan -ldl -lpthread -lX11 -lXxf86vm -lXrandr -lXi
CXXFLAGS = -std=c++17 -I. -I$(CURDIR)/include -I$(VULKAN_SDK_PATH)/include

#Linking glfw statically,currently off
links = $(wildcard lib/*.a)

#Directory information
BIN_DIR = bin
OBJ_DIR = obj
SHADER_DIR = shaders

# create list of all spv files and set as dependency
vertSources = $(shell find ./shaders -type f -name "*.vert")
vertObjFiles = $(patsubst %.vert, %.vert.spv, $(vertSources))
fragSources = $(shell find ./shaders -type f -name "*.frag")
fragObjFiles = $(patsubst %.frag, %.frag.spv, $(fragSources))

#source and object files
SRC_FILES = $(wildcard src/*.cpp)
OBJECTS_FILES = $(addprefix $(OBJ_DIR)/, $(notdir $(SRC_FILES:.cpp=.o)))

# Shaders should be compiled so program can read from thier object files
$(project):$(vertObjFiles) $(fragObjFiles) $(OBJECTS_FILES) 
	$(CXX) $(OBJECTS_FILES) $(CXXFLAGS) $(LDFLAGS) -o $(BIN_DIR)/$(project)

$(OBJ_DIR)/%.o: src/%.cpp
	$(CXX) -c $(CXXFLAGS) $< -o $@
%.spv: %
	glslangValidator -V $< -o $@

clean:
	-@rm $(BIN_DIR)/$(project)
	-@rm $(OBJ_DIR)/*.o
	-@rm $(SHADER_DIR)/*.spv