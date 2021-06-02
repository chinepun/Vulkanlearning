CXX = g++ -O3
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
SHADER_VERT_OBJ_DIR = shaderobj/vertex
SHADER_FRAG_OBJ_DIR = shaderobj/fragment

#source and object files
SRC_FILES = $(wildcard src/*.cpp)
OBJECTS_FILES = $(addprefix $(OBJ_DIR)/, $(notdir $(SRC_FILES:.cpp=.o)))
VERTEX_SHADERS = $(wildcard shaders/*.vert)
FRAGMENT_SHADERS = $(wildcard shaders/*.frag)
SHADER_VERT_OBJ_FILES = $(addprefix $(SHADER_VERT_OBJ_DIR)/, $(notdir $(VERTEX_SHADERS:.vert=.vert.spv)))
SHADER_FRAG_OBJ_FILES = $(addprefix $(SHADER_FRAG_OBJ_DIR)/, $(notdir $(FRAGMENT_SHADERS:.frag=.frag.spv)))

# Shaders should be compiled so program can read from thier object files
$(project):$(SHADER_VERT_OBJ_FILES) $(SHADER_FRAG_OBJ_FILES) $(OBJECTS_FILES)
	$(CXX) $(OBJECTS_FILES) $(CXXFLAGS) $(LDFLAGS) -o $(BIN_DIR)/$(project)

$(OBJ_DIR)/%.o: src/%.cpp
	$(CXX) -c $(CXXFLAGS) $< -o $@

$(SHADER_VERT_OBJ_DIR)/%.vert.spv: shaders/%.vert
	glslangValidator -V $< -o $@
$(SHADER_FRAG_OBJ_DIR)/%.frag.spv: shaders/%.frag
	glslangValidator -V $< -o $@

clean:
	-@rm $(BIN_DIR)/$(project)
	-@rm $(OBJ_DIR)/*.o
	-@rm $(SHADER_VERT_OBJ_DIR)/*.spv
	-@rm $(SHADER_FRAG_OBJ_DIR)/*.spv