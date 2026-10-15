build:
	@gcc main.c src/*.c \
		-I include -o model.exe -O3 -lm
