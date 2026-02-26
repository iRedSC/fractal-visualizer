import { Application, Mesh, Geometry, Shader, GlProgram } from 'pixi.js';
const app = new Application();
app.init().then(() => {
    const glProgram = GlProgram.from({
        vertex: "in vec2 aPosition; void main() { gl_Position = vec4(aPosition, 0.0, 1.0); }",
        fragment: "out vec4 finalColor; uniform vec4 uBounds; void main() { finalColor = uBounds; }"
    });
    const shader = new Shader({
        glProgram,
        resources: {
            uBounds: [1, 0, 0, 1]
        }
    });
    console.log("Success");
    process.exit(0);
}).catch(console.error);
