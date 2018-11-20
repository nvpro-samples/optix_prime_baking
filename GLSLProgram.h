/* Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */
 
 // Simple class to contain GLSL shaders/programs

#ifndef GLSL_PROGRAM_H
#define GLSL_PROGRAM_H

#include <include_gl.h>
#include <stdio.h>

class GLSLProgram
{
public:
    // construct program from strings
    GLSLProgram(const char*progName=NULL);
	GLSLProgram(const char *vsource, const char *fsource);
    GLSLProgram(const char *vsource, const char *gsource, const char *fsource,
                GLenum gsInput = GL_POINTS, GLenum gsOutput = GL_TRIANGLE_STRIP, int maxVerts=4);

	~GLSLProgram();

	void enable();
	void disable();

	void setUniform1f(const GLchar *name, GLfloat x);
	void setUniform2f(const GLchar *name, GLfloat x, GLfloat y);
    void setUniform2fv(const GLchar *name, float *v) { setUniformfv(name, v, 2, 1); }
    void setUniform3f(const GLchar *name, float x, float y, float z);
    void setUniform3fv(const GLchar *name, float *v) { setUniformfv(name, v, 3, 1); }
    void setUniform4f(const GLchar *name, float x, float y=0.0f, float z=0.0f, float w=0.0f);
	void setUniformfv(const GLchar *name, GLfloat *v, int elementSize, int count=1);
    void setUniformMatrix4fv(const GLchar *name, GLfloat *m, bool transpose);

	void setUniform1i(const GLchar *name, GLint x);
	void setUniform2i(const GLchar *name, GLint x, GLint y);
    void setUniform3i(const GLchar *name, int x, int y, int z);

	void bindTexture(const GLchar *name, GLuint tex, GLenum target, GLint unit);
	void bindImage  (const GLchar *name, GLint unit, GLuint tex, GLint level, GLboolean layered, GLint layer, GLenum access, GLenum format);

	inline GLuint getProgId() { return mProg; }
	
    GLuint compileProgram(const char *vsource, const char *gsource, const char *fsource,
                          GLenum gsInput = GL_POINTS, GLenum gsOutput = GL_TRIANGLE_STRIP, int maxVerts=4);
    GLuint compileProgramFromFiles(const char *vFilename,  const char *gFilename, const char *fFilename,
                       GLenum gsInput = GL_POINTS, GLenum gsOutput = GL_TRIANGLE_STRIP, int maxVerts=4);
    void setShaderNames(const char*ProgName, const char *VSName=NULL,const char *GSName=NULL,const char *FSName=NULL);
    static bool setIncludeFromFile(const char *includeName, const char* filename);
    static void setIncludeFromString(const char *includeName, const char* str);
private:
    static char *readTextFile(const char *filename);
    char *curVSName, *curFSName, *curGSName, *curProgName;

	GLuint mProg;
    static char* incPaths[];
};

#endif