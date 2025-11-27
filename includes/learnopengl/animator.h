// Matt edit the code to support cross fade blending of 2 clips
// 11/2/2024


#pragma once
#include <glm/glm.hpp>
#define GLM_ENABLE_EXPERIMENTAL
#include <glm/gtc/quaternion.hpp>
#include <glm/gtx/quaternion.hpp>   
#include <glm/gtx/norm.hpp>         

#include <glm/glm.hpp>
#include <map>
#include <vector>
#include <assimp/scene.h>
#include <assimp/Importer.hpp>
#include <learnopengl/animation.h>
#include <learnopengl/bone.h>

#include <string>
#include <unordered_map>

class Animator
{
public:
	Animator(Animation* animation)
	{
		m_CurrentTime = 0.0;
		m_CurrentAnimation = animation;
		m_CurrentAnimation2 = NULL;
		m_blendAmount = 0;

		m_FinalBoneMatrices.reserve(100);

		for (int i = 0; i < 100; i++)
			m_FinalBoneMatrices.push_back(glm::mat4(1.0f));
	}

	void UpdateAnimation(float dt)
	{
		m_DeltaTime = dt;
		if (m_CurrentAnimation)
		{
			m_CurrentTime += m_CurrentAnimation->GetTicksPerSecond() * dt;
			m_CurrentTime = fmod(m_CurrentTime, m_CurrentAnimation->GetDuration());

			if (m_CurrentAnimation2)
			{
				m_CurrentTime2 += m_CurrentAnimation2->GetTicksPerSecond() * dt;
				m_CurrentTime2 = fmod(m_CurrentTime2, m_CurrentAnimation2->GetDuration());
			}

			CalculateBoneTransform(&m_CurrentAnimation->GetRootNode(), glm::mat4(1.0f));
		}
	}

	void PlayAnimation(Animation* pAnimation, Animation* pAnimation2, float time1, float time2, float blend)
	{
		m_CurrentAnimation = pAnimation;
		m_CurrentTime = time1;
		m_CurrentAnimation2 = pAnimation2;
		m_CurrentTime2 = time2;
		m_blendAmount = blend;
	}

	void SetBoneOverride(const std::string& boneName, const glm::mat4& overrideTransform, float weight)
	{
		if (weight <= 0.0f) {
			m_BoneOverrides.erase(boneName);
			return;
		}
		BoneOverride bo;
		bo.transform = overrideTransform;
		bo.weight = glm::clamp(weight, 0.0f, 1.0f);
		m_BoneOverrides[boneName] = bo;
	}

	void ClearBoneOverride(const std::string& boneName)
	{
		m_BoneOverrides.erase(boneName);
	}

	void ClearAllOverrides()
	{
		m_BoneOverrides.clear();
	}

	glm::mat4 UpdateBlend(Bone* Bone1, Bone* Bone2) {
		glm::vec3 bonePos1, bonePos2, finalPos;
		glm::vec3 boneScale1, boneScale2, finalScale;
		glm::quat boneRot1, boneRot2, finalRot;

		Bone1->InterpolatePosition(m_CurrentTime, bonePos1);
		Bone2->InterpolatePosition(m_CurrentTime2, bonePos2);
		Bone1->InterpolateRotation(m_CurrentTime, boneRot1);
		Bone2->InterpolateRotation(m_CurrentTime2, boneRot2);
		Bone1->InterpolateScaling(m_CurrentTime, boneScale1);
		Bone2->InterpolateScaling(m_CurrentTime2, boneScale2);

		finalPos = glm::mix(bonePos1, bonePos2, m_blendAmount);
		finalRot = glm::slerp(boneRot1, boneRot2, m_blendAmount);
		finalRot = glm::normalize(finalRot);
		finalScale = glm::mix(boneScale1, boneScale2, m_blendAmount);

		glm::mat4 translation = glm::translate(glm::mat4(1.0f), finalPos);
		glm::mat4 rotation = glm::toMat4(finalRot);
		glm::mat4 scale = glm::scale(glm::mat4(1.0f), finalScale);

		glm::mat4 TRS = glm::mat4(1.0f);
		TRS = translation * rotation * scale;
		return TRS;
	}

	void CalculateBoneTransform(const AssimpNodeData* node, glm::mat4 parentTransform)
	{
		std::string nodeName = node->name;
		glm::mat4 nodeTransform = node->transformation;

		Bone* Bone1 = m_CurrentAnimation->FindBone(nodeName);
		Bone* Bone2 = NULL;
		if (m_CurrentAnimation2) {
			Bone2 = m_CurrentAnimation2->FindBone(nodeName);
		}

		if (Bone1)
		{
			Bone1->Update(m_CurrentTime);
			nodeTransform = Bone1->GetLocalTransform();

			if (Bone2) {
				nodeTransform = UpdateBlend(Bone1, Bone2);
			}
		}

		auto it = m_BoneOverrides.find(nodeName);
		if (it != m_BoneOverrides.end())
		{
			const BoneOverride& bo = it->second;
			glm::vec3 t0, s0;
			glm::quat r0;
			DecomposeTransform(nodeTransform, t0, r0, s0);

			glm::vec3 t1, s1;
			glm::quat r1;
			DecomposeTransform(bo.transform, t1, r1, s1);

			// Blend components
			glm::vec3 tBlend = glm::mix(t0, t1, bo.weight);
			glm::quat rBlend = glm::slerp(r0, r1, bo.weight);
			rBlend = glm::normalize(rBlend);
			glm::vec3 sBlend = glm::mix(s0, s1, bo.weight);

			nodeTransform = ComposeTransform(tBlend, rBlend, sBlend);
		}

		glm::mat4 globalTransformation = parentTransform * nodeTransform;

		auto boneInfoMap = m_CurrentAnimation->GetBoneIDMap();
		if (boneInfoMap.find(nodeName) != boneInfoMap.end())
		{
			int index = boneInfoMap[nodeName].id;
			glm::mat4 offset = boneInfoMap[nodeName].offset;
			m_FinalBoneMatrices[index] = globalTransformation * offset;
		}

		for (int i = 0; i < node->childrenCount; i++)
			CalculateBoneTransform(&node->children[i], globalTransformation);
	}

	std::vector<glm::mat4> GetFinalBoneMatrices()
	{
		return m_FinalBoneMatrices;
	}

	//private:
	std::vector<glm::mat4> m_FinalBoneMatrices;
	Animation* m_CurrentAnimation;
	Animation* m_CurrentAnimation2;
	float m_CurrentTime;
	float m_CurrentTime2;
	float m_DeltaTime;
	float m_blendAmount;

	// ---- per-bone overrides
	struct BoneOverride {
		glm::mat4 transform;
		float weight;
	};
	std::unordered_map<std::string, BoneOverride> m_BoneOverrides;


	// ---- helpers to decompose & compose TRS
	static void DecomposeTransform(const glm::mat4& mat, glm::vec3& outTranslation, glm::quat& outRotation, glm::vec3& outScale)
	{
		// translation
		outTranslation = glm::vec3(mat[3]);

		// extract scale from columns
		glm::vec3 col0 = glm::vec3(mat[0]);
		glm::vec3 col1 = glm::vec3(mat[1]);
		glm::vec3 col2 = glm::vec3(mat[2]);
		float sx = glm::length(col0);
		float sy = glm::length(col1);
		float sz = glm::length(col2);
		outScale = glm::vec3(sx, sy, sz);

		// remove scale from rotation matrix
		glm::mat3 rotMat(1.0f);
		if (sx != 0.0f) rotMat[0] = col0 / sx;
		if (sy != 0.0f) rotMat[1] = col1 / sy;
		if (sz != 0.0f) rotMat[2] = col2 / sz;

		outRotation = glm::quat_cast(rotMat);
		outRotation = glm::normalize(outRotation);
	}

	static glm::mat4 ComposeTransform(const glm::vec3& translation, const glm::quat& rotation, const glm::vec3& scale)
	{
		glm::mat4 T = glm::translate(glm::mat4(1.0f), translation);
		glm::mat4 R = glm::toMat4(rotation);
		glm::mat4 S = glm::scale(glm::mat4(1.0f), scale);
		return T * R * S;
	}
};
