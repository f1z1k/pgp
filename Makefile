CC= nvcc
GLFLAGS= -lGLEW -lGLU -lGL -lglut -Werror cross-execution-space-call -lcurand
LDFLAGS= -arch=sm_20
SRC= main.cu
 
all: kp test

run: kp
	optirun ./kp &
clean: kp
	rm ./kp
update:
	sudo update-alternatives --config x86_64-linux-gnu_gl_conf
kp: $(SRC)
	$(CC) $(GLFLAGS) $(LDFLAGS) -g $< -o $@

test: test.cu
	$(CC) $(GLFLAGS) $(LDFLAGS) -g $< -o $@
