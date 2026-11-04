pluginManagement {
    repositories {
        google()
        mavenCentral()
        gradlePluginPortal()
    }
}
dependencyResolutionManagement {
    repositoriesMode.set(RepositoriesMode.FAIL_ON_PROJECT_REPOS)
    repositories {
        google()
        // The Wearables Device Access Toolkit is published on Maven Central (com.meta.wearable).
        mavenCentral()
    }
}
rootProject.name = "glasses-guide"
include(":app")
