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
// recursively print animation node hierarchy names
void PrintNodeHierarchy(const AssimpNodeData& node, int depth = 0) {
	for (int i = 0; i < depth; ++i) std::cout << "  ";
	std::cout << node.name << "\n";
	for (int i = 0; i < node.childrenCount; ++i)
		PrintNodeHierarchy(node.children[i], depth + 1);
}

// print keys from the boneIDMap
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

// settings
const unsigned int SCR_WIDTH = 1000;
const unsigned int SCR_HEIGHT = 800;

// camera
Camera camera(glm::vec3(0.0f, 0.0f, 3.0f));
float lastX = SCR_WIDTH / 2.0f;
float lastY = SCR_HEIGHT / 2.0f;
bool firstMouse = true;

// timing
float deltaTime = 0.0f;
float lastFrame = 0.0f;

bool holdGun = false;                
bool eWasDown = false;                
float holdProgress = 0.0f;              
const float holdSpeed = 6.0f;        
std::string rightHandBoneName = "RightHand";

enum AnimState {
	IDLE = 1,
	IDLE_PUNCH,
	PUNCH_IDLE,
	IDLE_KICK,
	KICK_IDLE,
	IDLE_WALK,
	WALK_IDLE,
	WALK,
	EQUIP,        // standing, holding weapon
	EQUIP_WALK,   // walking while holding
	EQUIP_IDLE    // transition back to equip idle (if you need)
};

int main()
{
	// glfw: initialize and configure
	// ------------------------------
	glfwInit();
	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
	glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

#ifdef __APPLE__
	glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
#endif

	// glfw window creation
	// --------------------
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

	// tell GLFW to capture our mouse
	glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);

	// glad: load all OpenGL function pointers
	// ---------------------------------------
	if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress))
	{
		std::cout << "Failed to initialize GLAD" << std::endl;
		return -1;
	}

	// tell stb_image.h to flip loaded texture's on the y-axis (before loading model).
	stbi_set_flip_vertically_on_load(true);

	// configure global opengl state
	// -----------------------------
	glEnable(GL_DEPTH_TEST);

	// build and compile shaders
	// -------------------------
	Shader ourShader("anim_model.vs", "anim_model.fs");

	
	// load models
	// -----------
	// idle 3.3, walk 2.06, run 0.83, punch 1.03, kick 1.6
	Model ourModel(FileSystem::getPath("resources/objects/mixamo/Boss.dae"));
	
	Animation idleAnimation(FileSystem::getPath("resources/objects/mixamo/idle.dae"), &ourModel);

	std::string chosen = "mixamorig_RightForeArm";

	rightHandBoneName = chosen;
	std::cout << "Using bone: " << rightHandBoneName << "\n";
	Bone* testBone = idleAnimation.FindBone(rightHandBoneName);
	if (testBone) {
		std::cout << "[OK] idleAnimation contains bone '" << rightHandBoneName << "' (id=" << testBone->GetBoneID() << ")\n";
	}
	else {
		std::cout << "[WARN] idleAnimation does NOT contain bone '" << rightHandBoneName << "'.\n";
	}


	// DEBUG: print bone map & hierarchy for idle animation
	auto boneMapIdle = idleAnimation.GetBoneIDMap();
	PrintBoneIDMap(boneMapIdle);
	std::cout << "Animation root node hierarchy:\n";
	PrintNodeHierarchy(idleAnimation.GetRootNode());

	// try to auto-detect right-hand bone name from the boneMap keys (case-insensitive substring)
	auto pickHandBone = [&](const std::map<std::string, BoneInfo>& bm) -> std::string {
		std::vector<std::string> candidates;
		for (auto& kv : bm) {
			std::string key = kv.first;
			std::string lower = key;
			std::transform(lower.begin(), lower.end(), lower.begin(), ::tolower);
			if (lower.find("righthand") != std::string::npos ||
				lower.find("right_hand") != std::string::npos ||
				lower.find("hand_r") != std::string::npos ||
				lower.find("mixamorig:right") != std::string::npos ||
				lower.find("r_hand") != std::string::npos ||
				lower.find("right") != std::string::npos && lower.find("hand") != std::string::npos) {
				candidates.push_back(key);
			}
		}
		if (!candidates.empty()) return candidates[0];
		// fallback: try any key containing "hand"
		for (auto& kv : bm) {
			std::string lower = kv.first; std::transform(lower.begin(), lower.end(), lower.begin(), ::tolower);
			if (lower.find("hand") != std::string::npos) return kv.first;
		}
		return std::string(); // not found
		};

	std::string autoRight = pickHandBone(boneMapIdle);
	if (!autoRight.empty()) {
		std::cout << "[AUTO] picking right-hand bone name: '" << autoRight << "'\n";
		rightHandBoneName = autoRight;
	}
	else {
		std::cout << "[AUTO] right-hand bone not found automatically. Pick a name from the BoneIDMap above and set rightHandBoneName accordingly.\n";
	}
	Animation walkAnimation(FileSystem::getPath("resources/objects/mixamo/walk.dae"), &ourModel);
	Animation runAnimation(FileSystem::getPath("resources/objects/mixamo/run.dae"), &ourModel);
	Animation punchAnimation(FileSystem::getPath("resources/objects/mixamo/punch.dae"), &ourModel);
	Animation kickAnimation(FileSystem::getPath("resources/objects/mixamo/kick.dae"), &ourModel);
	Animator animator(&idleAnimation);
	enum AnimState charState = IDLE;
	float blendAmount = 0.0f;
	float blendRate = 0.055f;

	// draw in wireframe
	//glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);

	// render loop
	// -----------
	while (!glfwWindowShouldClose(window))
	{
		// per-frame time logic
		// --------------------
		float currentFrame = glfwGetTime();
		deltaTime = currentFrame - lastFrame;
		lastFrame = currentFrame;

		// input
		// -----
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
				// only punch if not holding a gun
				blendAmount = 0.0f;
				animator.PlayAnimation(&idleAnimation, &punchAnimation, animator.m_CurrentTime, 0.0f, blendAmount);
				charState = IDLE_PUNCH;
			}
			else if (glfwGetKey(window, GLFW_KEY_K) == GLFW_PRESS) {
				// only kick if not holding a gun
				blendAmount = 0.0f;
				animator.PlayAnimation(&idleAnimation, &kickAnimation, animator.m_CurrentTime, 0.0f, blendAmount);
				charState = IDLE_KICK;
			}
			printf("idle \n");
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
			printf("idle_walk \n");
			break;

		case WALK:
			animator.PlayAnimation(&walkAnimation, NULL, animator.m_CurrentTime, animator.m_CurrentTime2, blendAmount);
			if (glfwGetKey(window, GLFW_KEY_UP) != GLFW_PRESS) {
				charState = WALK_IDLE;
			}
			printf("walking\n");
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
			printf("walk_idle \n");
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
			printf("idle_punch\n");
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
				printf("punch_idle \n");
			}
			else {
				printf("punching \n");
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
			printf("idle_kick\n");
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
				printf("kick_idle \n");
			}
			else {
				printf("kicking \n");
			}
			break;

			// ---------- equip states ----------
		case EQUIP:
			// If moving forward while equipped, go to equip-walk
			if (glfwGetKey(window, GLFW_KEY_UP) == GLFW_PRESS) {
				blendAmount = 0.0f;
				animator.PlayAnimation(&idleAnimation, &walkAnimation, animator.m_CurrentTime, animator.m_CurrentTime2, blendAmount);
				charState = EQUIP_WALK;
			}
			// while in EQUIP we *don't* start punch/kick from idle (you can implement shooting or different anims)
			printf("equip_idle (holding)\n");
			break;

		case EQUIP_WALK:
			// Walk animation plays but we remain in equip-walk until key release.
			blendAmount += blendRate;
			blendAmount = fmod(blendAmount, 1.0f);
			animator.PlayAnimation(&idleAnimation, &walkAnimation, animator.m_CurrentTime, animator.m_CurrentTime2, blendAmount);
			if (glfwGetKey(window, GLFW_KEY_UP) != GLFW_PRESS) {
				// return to equip idle
				blendAmount = 0.0f;
				float startTime = animator.m_CurrentTime2;
				animator.PlayAnimation(&idleAnimation, NULL, startTime, 0.0f, blendAmount);
				charState = EQUIP;
			}
			printf("equip walking\n");
			break;

		default:
			break;
		}
		// handle E toggle (edge detect)
		bool eDown = (glfwGetKey(window, GLFW_KEY_E) == GLFW_PRESS);
		if (eDown && !eWasDown) {
			holdGun = !holdGun;
			if (holdGun) {
				// Enter equip state
				charState = EQUIP;
			}
			else {
				// Exit equip -> go back to idle
				charState = IDLE;
			}
		}
		eWasDown = eDown;

		// Update holdProgress but treat equip states as target = 1.0
		bool inEquipState = (charState == EQUIP || charState == EQUIP_WALK || charState == EQUIP_IDLE);
		if (inEquipState) holdProgress += holdSpeed * deltaTime;
		else             holdProgress -= holdSpeed * deltaTime;
		holdProgress = glm::clamp(holdProgress, 0.0f, 1.0f);


		// update animation
		animator.UpdateAnimation(deltaTime);

		

		
		// render
		// ------
		glClearColor(0.05f, 0.05f, 0.05f, 1.0f);
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

		// don't forget to enable shader before setting uniforms
		ourShader.use();

		// view/projection transformations
		glm::mat4 projection = glm::perspective(glm::radians(camera.Zoom), (float)SCR_WIDTH / (float)SCR_HEIGHT, 0.1f, 100.0f);
		glm::mat4 view = camera.GetViewMatrix();
		ourShader.setMat4("projection", projection);
		ourShader.setMat4("view", view);

		rightHandBoneName = "mixamorig_RightForeArm";

		glm::mat4 targetExtra = glm::translate(glm::mat4(1.0f), glm::vec3(0.0f, 0.06f, 0.0f))
			* glm::rotate(glm::mat4(1.0f), glm::radians(-35.0f), glm::vec3(1.0f, 0.0f, 0.0f));
		animator.SetBoneOverride(rightHandBoneName, targetExtra, holdProgress);

		// set override on the animator (transform is local bone transform to blend toward, weight is holdProgress)
		animator.SetBoneOverride(rightHandBoneName, targetExtra, holdProgress);

		// quick verify: print final bone translation for this bone
		auto finalMats = animator.GetFinalBoneMatrices();
		auto boneMap = idleAnimation.GetBoneIDMap();
		auto it = boneMap.find(rightHandBoneName);
		if (it != boneMap.end()) {
			int idx = it->second.id;
			if (idx >= 0 && idx < (int)finalMats.size()) {
				glm::vec3 tr = glm::vec3(finalMats[idx][3]);
				std::cout << "[FINAL] bone '" << rightHandBoneName << "' idx=" << idx
					<< " translation=(" << tr.x << "," << tr.y << "," << tr.z << ") holdProg=" << holdProgress << "\n";
			}
		}


        auto transforms = animator.GetFinalBoneMatrices();
		for (int i = 0; i < transforms.size(); ++i)
			ourShader.setMat4("finalBonesMatrices[" + std::to_string(i) + "]", transforms[i]);


		// render the loaded model
		glm::mat4 model = glm::mat4(1.0f);
		model = glm::translate(model, glm::vec3(0.0f, -0.4f, 0.0f)); // translate it down so it's at the center of the scene
		model = glm::scale(model, glm::vec3(.5f, .5f, .5f));	// it's a bit too big for our scene, so scale it down
		ourShader.setMat4("model", model);
		ourModel.Draw(ourShader);


		// glfw: swap buffers and poll IO events (keys pressed/released, mouse moved etc.)
		// -------------------------------------------------------------------------------
		glfwSwapBuffers(window);
		glfwPollEvents();
	}

	// glfw: terminate, clearing all previously allocated GLFW resources.
	// ------------------------------------------------------------------
	glfwTerminate();
	return 0;
}

// process all input: query GLFW whether relevant keys are pressed/released this frame and react accordingly
// ---------------------------------------------------------------------------------------------------------
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
// ---------------------------------------------------------------------------------------------
void framebuffer_size_callback(GLFWwindow* window, int width, int height)
{
	// make sure the viewport matches the new window dimensions; note that width and 
	// height will be significantly larger than specified on retina displays.
	glViewport(0, 0, width, height);
}

// glfw: whenever the mouse moves, this callback is called
// -------------------------------------------------------
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

// glfw: whenever the mouse scroll wheel scrolls, this callback is called
// ----------------------------------------------------------------------
void scroll_callback(GLFWwindow* window, double xoffset, double yoffset)
{
	camera.ProcessMouseScroll(yoffset);
}
