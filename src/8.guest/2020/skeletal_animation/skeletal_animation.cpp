
#include <glad/glad.h>
#include <GLFW/glfw3.h>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#define GLM_ENABLE_EXPERIMENTAL
#include <glm/gtc/quaternion.hpp>
#include <glm/gtx/quaternion.hpp>

#include <learnopengl/filesystem.h>
#include <learnopengl/shader_m.h>
#include <learnopengl/camera.h>
#include <learnopengl/animator.h>
#include <learnopengl/model_animation.h>

#include <iostream>
#include <algorithm>

void PrintNodeHierarchy(const AssimpNodeData& node, int depth = 0) {
	for (int i = 0; i < depth; ++i) std::cout << "  ";
	std::cout << node.name << "\n";
	for (int i = 0; i < node.childrenCount; ++i)
		PrintNodeHierarchy(node.children[i], depth + 1);
}

void PrintBoneIDMap(const std::map<std::string, BoneInfo>& map) {
	std::cout << "BoneIDMap keys (" << map.size() << "):\n";
	for (auto& kv : map) {
		std::cout << "  '" << kv.first << "' id=" << kv.second.id << "\n";
	}
}

static inline void PlayIfDifferent(Animator& anim, Animation* newAnim) {
	if (anim.m_CurrentAnimation != newAnim) {
		anim.PlayAnimation(newAnim, NULL, anim.m_CurrentTime, 0.0f, 0.0f);
	}
}

void framebuffer_size_callback(GLFWwindow* window, int width, int height);
void mouse_callback(GLFWwindow* window, double xpos, double ypos);
void scroll_callback(GLFWwindow* window, double xoffset, double yoffset);
void processInput(GLFWwindow* window);


const unsigned int SCR_WIDTH = 1000;
const unsigned int SCR_HEIGHT = 800;


Camera camera(glm::vec3(0.0f, 0.0f, 3.0f));
float lastX = SCR_WIDTH / 2.0f;
float lastY = SCR_HEIGHT / 2.0f;
bool firstMouse = true;

float deltaTime = 0.0f;
float lastFrame = 0.0f;

bool eWasDown = false;
bool isEquipped = false;

enum AnimState {
	IDLE = 1,
	IDLE_PUNCH,
	PUNCH_IDLE,
	IDLE_KICK,
	KICK_IDLE,
	IDLE_WALK,
	WALK_IDLE,
	WALK,
	EQUIP
};

int main()
{
	glfwInit();
	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
	glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
#ifdef __APPLE__
	glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
#endif

	GLFWwindow* window = glfwCreateWindow(SCR_WIDTH, SCR_HEIGHT, "LearnOpenGL", NULL, NULL);
	if (window == NULL)
	{
		std::cout << "Failed to create GLFW window" << std::endl;
		glfwTerminate();
		return -1;
	}
	glfwMakeContextCurrent(window);
	glfwSetFramebufferSizeCallback(window, framebuffer_size_callback);
	glfwSetCursorPosCallback(window, mouse_callback);
	glfwSetScrollCallback(window, scroll_callback);
	glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);

	if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress))
	{
		std::cout << "Failed to initialize GLAD" << std::endl;
		return -1;
	}

	stbi_set_flip_vertically_on_load(true);
	glEnable(GL_DEPTH_TEST);

	Shader ourShader("anim_model.vs", "anim_model.fs");

	Model ourModel(FileSystem::getPath("resources/objects/mixamo/Boss.dae"));

	Animation idleAnimation(FileSystem::getPath("resources/objects/mixamo/idle.dae"), &ourModel);
	Animation walkAnimation(FileSystem::getPath("resources/objects/mixamo/walk.dae"), &ourModel);
	Animation runAnimation(FileSystem::getPath("resources/objects/mixamo/run.dae"), &ourModel);
	Animation punchAnimation(FileSystem::getPath("resources/objects/mixamo/punch.dae"), &ourModel);
	Animation kickAnimation(FileSystem::getPath("resources/objects/mixamo/kick.dae"), &ourModel);
	Animation equipAnimation(FileSystem::getPath("resources/objects/mixamo/equip.dae"), &ourModel);

	auto boneMapIdle = idleAnimation.GetBoneIDMap();
	PrintBoneIDMap(boneMapIdle);
	std::cout << "Animation root node hierarchy:\n";
	PrintNodeHierarchy(idleAnimation.GetRootNode());

	Animator animator(&idleAnimation);
	enum AnimState charState = IDLE;
	float blendAmount = 0.0f;
	float blendRate = 0.055f;

	Model glockModel("../../../../resources/objects/pistol/Glock18.dae");
	std::cout << "[GLOCK] meshes=" << glockModel.meshes.size() << "\n";
	if (glockModel.meshes.empty()) {
		std::cout << "[GLOCK] ERROR: no meshes loaded. Check path: resources/objects/pistol/Glock18.dae\n";
	}

	glm::vec3 gmin(FLT_MAX), gmax(-FLT_MAX);
	for (auto& m : glockModel.meshes) {
		for (auto& v : m.vertices) {
			gmin = glm::min(gmin, v.Position);
			gmax = glm::max(gmax, v.Position);
		}
	}
	glm::vec3 gsize = gmax - gmin;
	float gmaxEdge = std::max(std::max(gsize.x, gsize.y), gsize.z);
	float pistolNormalizeScale = (gmaxEdge > 0.0f) ? (0.08f / gmaxEdge) : 1.0f; 

	PlayIfDifferent(animator, &idleAnimation);

	while (!glfwWindowShouldClose(window))
	{
		float currentFrame = glfwGetTime();
		deltaTime = currentFrame - lastFrame;
		lastFrame = currentFrame;

		processInput(window);
		if (glfwGetKey(window, GLFW_KEY_1) == GLFW_PRESS)
			PlayIfDifferent(animator, &idleAnimation);
		if (glfwGetKey(window, GLFW_KEY_2) == GLFW_PRESS)
			PlayIfDifferent(animator, &walkAnimation);
		if (glfwGetKey(window, GLFW_KEY_3) == GLFW_PRESS)
			PlayIfDifferent(animator, &punchAnimation);
		if (glfwGetKey(window, GLFW_KEY_4) == GLFW_PRESS)
			PlayIfDifferent(animator, &kickAnimation);

		switch (charState) {
		case IDLE:
			if (glfwGetKey(window, GLFW_KEY_UP) == GLFW_PRESS) {
				blendAmount = 0.0f;
				animator.PlayAnimation(&idleAnimation, &walkAnimation, animator.m_CurrentTime, 0.0f, blendAmount);
				charState = IDLE_WALK;
			}
			else if (glfwGetKey(window, GLFW_KEY_J) == GLFW_PRESS) {
				blendAmount = 0.0f;
				animator.PlayAnimation(&idleAnimation, &punchAnimation, animator.m_CurrentTime, 0.0f, blendAmount);
				charState = IDLE_PUNCH;
			}
			else if (glfwGetKey(window, GLFW_KEY_K) == GLFW_PRESS) {
				blendAmount = 0.0f;
				animator.PlayAnimation(&idleAnimation, &kickAnimation, animator.m_CurrentTime, 0.0f, blendAmount);
				charState = IDLE_KICK;
			}
			break;

		case IDLE_WALK:
			blendAmount += blendRate;
			blendAmount = fmod(blendAmount, 1.0f);
			animator.PlayAnimation(&idleAnimation, &walkAnimation, animator.m_CurrentTime, animator.m_CurrentTime2, blendAmount);
			if (blendAmount > 0.9f) {
				blendAmount = 0.0f;
				float startTime = animator.m_CurrentTime2;
				animator.PlayAnimation(&walkAnimation, NULL, startTime, 0.0f, blendAmount);
				charState = WALK;
			}
			break;

		case WALK:
			animator.PlayAnimation(&walkAnimation, NULL, animator.m_CurrentTime, animator.m_CurrentTime2, blendAmount);
			if (glfwGetKey(window, GLFW_KEY_UP) != GLFW_PRESS) {
				charState = WALK_IDLE;
			}
			break;

		case WALK_IDLE:
			blendAmount += blendRate;
			blendAmount = fmod(blendAmount, 1.0f);
			animator.PlayAnimation(&walkAnimation, &idleAnimation, animator.m_CurrentTime, animator.m_CurrentTime2, blendAmount);
			if (blendAmount > 0.9f) {
				blendAmount = 0.0f;
				float startTime = animator.m_CurrentTime2;
				animator.PlayAnimation(&idleAnimation, NULL, startTime, 0.0f, blendAmount);
				charState = IDLE;
			}
			break;

		case IDLE_PUNCH:
			blendAmount += blendRate;
			blendAmount = fmod(blendAmount, 1.0f);
			animator.PlayAnimation(&idleAnimation, &punchAnimation, animator.m_CurrentTime, animator.m_CurrentTime2, blendAmount);
			if (blendAmount > 0.9f) {
				blendAmount = 0.0f;
				float startTime = animator.m_CurrentTime2;
				animator.PlayAnimation(&punchAnimation, NULL, startTime, 0.0f, blendAmount);
				charState = PUNCH_IDLE;
			}
			break;

		case PUNCH_IDLE:
			if (animator.m_CurrentTime > 0.7f) {
				blendAmount += blendRate;
				blendAmount = fmod(blendAmount, 1.0f);
				animator.PlayAnimation(&punchAnimation, &idleAnimation, animator.m_CurrentTime, animator.m_CurrentTime2, blendAmount);
				if (blendAmount > 0.9f) {
					blendAmount = 0.0f;
					float startTime = animator.m_CurrentTime2;
					animator.PlayAnimation(&idleAnimation, NULL, startTime, 0.0f, blendAmount);
					charState = IDLE;
				}
			}
			break;

		case IDLE_KICK:
			blendAmount += blendRate;
			blendAmount = fmod(blendAmount, 1.0f);
			animator.PlayAnimation(&idleAnimation, &kickAnimation, animator.m_CurrentTime, animator.m_CurrentTime2, blendAmount);
			if (blendAmount > 0.9f) {
				blendAmount = 0.0f;
				float startTime = animator.m_CurrentTime2;
				animator.PlayAnimation(&kickAnimation, NULL, startTime, 0.0f, blendAmount);
				charState = KICK_IDLE;
			}
			break;

		case KICK_IDLE:
			if (animator.m_CurrentTime > 1.0f) {
				blendAmount += blendRate;
				blendAmount = fmod(blendAmount, 1.0f);
				animator.PlayAnimation(&kickAnimation, &idleAnimation, animator.m_CurrentTime, animator.m_CurrentTime2, blendAmount);
				if (blendAmount > 0.9f) {
					blendAmount = 0.0f;
					float startTime = animator.m_CurrentTime2;
					animator.PlayAnimation(&idleAnimation, NULL, startTime, 0.0f, blendAmount);
					charState = IDLE;
				}
			}
			break;

		default:
			break;
		}

		bool eDown = (glfwGetKey(window, GLFW_KEY_E) == GLFW_PRESS);
		if (eDown && !eWasDown) {
			isEquipped = !isEquipped;
			if (isEquipped) {
				PlayIfDifferent(animator, &equipAnimation);
				charState = EQUIP;
			}
			else {
				PlayIfDifferent(animator, &idleAnimation);
				charState = IDLE;
			}
		}
		eWasDown = eDown;

		animator.UpdateAnimation(deltaTime);

		glClearColor(0.05f, 0.05f, 0.05f, 1.0f);
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

		ourShader.use();

		glm::mat4 projection = glm::perspective(glm::radians(camera.Zoom), (float)SCR_WIDTH / (float)SCR_HEIGHT, 0.1f, 100.0f);
		glm::mat4 view = camera.GetViewMatrix();
		ourShader.setMat4("projection", projection);
		ourShader.setMat4("view", view);

		auto transforms = animator.GetFinalBoneMatrices();
		for (int i = 0; i < transforms.size(); ++i)
			ourShader.setMat4("finalBonesMatrices[" + std::to_string(i) + "]", transforms[i]);

		glm::mat4 model = glm::mat4(1.0f);
		model = glm::translate(model, glm::vec3(0.0f, -0.4f, 0.0f));
		model = glm::scale(model, glm::vec3(.5f, .5f, .5f));
		ourShader.setMat4("model", model);
		ourModel.Draw(ourShader);
		auto finalTransforms = animator.GetFinalBoneMatrices();
		for (size_t i = 0; i < finalTransforms.size(); ++i) {
			ourShader.setMat4("finalBonesMatrices[" + std::to_string(i) + "]", finalTransforms[i]);
		}

		glm::mat4 modelMatrix = glm::mat4(1.0f);
		modelMatrix = glm::translate(modelMatrix, glm::vec3(0.0f, -0.4f, 0.0f));
		modelMatrix = glm::scale(modelMatrix, glm::vec3(.5f, .5f, .5f));
		ourShader.setMat4("model", modelMatrix);

		ourModel.Draw(ourShader);

		auto boneMap = idleAnimation.GetBoneIDMap();
		std::vector<std::string> tryNames = { "mixamorig_RightHand", "mixamorig_RightForeArm", "RightHand", "RightForeArm" };
		bool drawnPistol = false;

		for (const auto& bn : tryNames) {
			auto it = boneMap.find(bn);
			if (it == boneMap.end()) continue;

			int idx = it->second.id;
			if (idx < 0 || idx >= (int)finalTransforms.size()) continue;

			glm::mat4 bindPose = glm::inverse(it->second.offset);
			glm::mat4 boneGlobal = finalTransforms[idx] * bindPose;

			glm::mat4 pistolLocal = glm::mat4(1.0f);
			pistolLocal = glm::translate(pistolLocal, glm::vec3(0.02f, -0.02f, 0.06f));
			pistolLocal = glm::rotate(pistolLocal, glm::radians(180.0f), glm::vec3(0, 1, 0)); 
			pistolLocal = glm::rotate(pistolLocal, glm::radians(-10.0f), glm::vec3(1, 0, 0)); 
			pistolLocal = glm::scale(pistolLocal, glm::vec3(pistolNormalizeScale));

			glm::mat4 modelGlock = modelMatrix * boneGlobal * pistolLocal;
			ourShader.setMat4("model", modelGlock);
			glockModel.Draw(ourShader);

			drawnPistol = true;
			break;
		}
		if (!drawnPistol) {

		}

		glfwSwapBuffers(window);
		glfwPollEvents();
	}

	glfwTerminate();
	return 0;
}

void processInput(GLFWwindow* window)
{
	if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
		glfwSetWindowShouldClose(window, true);

	if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS)
		camera.ProcessKeyboard(FORWARD, deltaTime);
	if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS)
		camera.ProcessKeyboard(BACKWARD, deltaTime);
	if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS)
		camera.ProcessKeyboard(LEFT, deltaTime);
	if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS)
		camera.ProcessKeyboard(RIGHT, deltaTime);
}

// glfw: whenever the window size changed (by OS or user resize) this callback function executes
void framebuffer_size_callback(GLFWwindow* window, int width, int height)
{
	glViewport(0, 0, width, height);
}

// glfw: whenever the mouse moves, this callback is called
void mouse_callback(GLFWwindow* window, double xpos, double ypos)
{
	if (firstMouse)
	{
		lastX = xpos;
		lastY = ypos;
		firstMouse = false;
	}

	float xoffset = xpos - lastX;
	float yoffset = lastY - ypos; // reversed since y-coordinates go from bottom to top

	lastX = xpos;
	lastY = ypos;

	camera.ProcessMouseMovement(xoffset, yoffset);
}

// glfw: whenever the mouse scroll wheel scrolls, this callback function is called
void scroll_callback(GLFWwindow* window, double xoffset, double yoffset)
{
	camera.ProcessMouseScroll(yoffset);
}
