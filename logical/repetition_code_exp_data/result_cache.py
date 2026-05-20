import functools
import inspect
import pathlib
import pickle
from typing import Any, Callable, List, Optional, Union


class ResultCache:
    """Local result cache manager, supports use as decorator

    Used for caching data processing results, supports finding cache based on specified parameters.

    Example:
        >>> # Use as decorator
        >>> cache = ResultCache("./cache")
        >>>
        >>> @cache(key_params=["data_id", "version"])
        ... def process_data(data_id: str, version: int, temp_param: float):
        ...     # Time-consuming data processing
        ...     return result
        ...
        >>> result = process_data("exp1", 1, 0.5)  # First calculation and cache
        >>> result = process_data("exp1", 1, 0.8)  # Return cached result (temp_param is different but not in key_params)

        >>> # Use as regular cache
        >>> cache = ResultCache("./cache")
        >>> cache.set("my_key", {"data": [1, 2, 3]})
        >>> result = cache.get("my_key")
    """

    def __init__(self, cache_dir: Union[str, pathlib.Path]):
        """
        Args:
            cache_dir: Cache directory path
        """
        self.cache_dir = pathlib.Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _generate_cache_key(
        self,
        func: Callable,
        args: tuple,
        kwargs: dict,
        key_params: List[str],
    ) -> str:
        """Generate human-readable cache key"""
        # Get function signature
        sig = inspect.signature(func)
        bound_args = sig.bind(*args, **kwargs)
        bound_args.apply_defaults()

        # Build parameter dictionary
        all_params = dict(bound_args.arguments)

        # Only use specified key parameters
        filtered_params = {k: v for k, v in all_params.items() if k in key_params}

        # Build readable key name: func_name__param1=value1__param2=value2
        parts = [func.__name__]
        for value in filtered_params.values():
            parts.append(str(value))

        return self._sanitize_filename("__".join(parts))

    def _sanitize_filename(self, name: str, max_length: int = 200) -> str:
        """Clean string to make it suitable as filename"""
        # Windows/Linux invalid characters
        invalid_chars = '<>:"/\\|?*'
        for char in invalid_chars:
            name = name.replace(char, "_")

        # Truncate overly long strings
        if len(name) > max_length:
            # Keep front and back parts, use ellipsis in middle
            prefix = name[: max_length // 2]
            suffix = name[-max_length // 4 :]
            name = f"{prefix}...{suffix}"

        # Remove leading/trailing whitespace and dots (Windows cannot end with dot)
        name = name.strip().rstrip(".")

        # Handle empty string
        if not name:
            return "empty"

        return name

    def _get_cache_path(self, cache_key: str) -> pathlib.Path:
        """Get cache file path"""
        return self.cache_dir / f"{cache_key}.pkl"

    def _cache_key_exists(self, cache_key: str) -> bool:
        """Check if cache key exists by checking file existence"""
        cache_path = self._get_cache_path(cache_key)
        return cache_path.exists()

    def get(self, cache_key: str) -> Optional[Any]:
        """Get cache value"""
        cache_path = self._get_cache_path(cache_key)

        # Check if file exists
        if not cache_path.exists():
            return None

        # Load cache data
        try:
            with open(cache_path, "rb") as f:
                return pickle.load(f)
        except (pickle.PickleError, IOError, EOFError):
            try:
                cache_path.unlink()
            except OSError:
                pass
            return None

    def set(self, cache_key: str, value: Any):
        """Set cache value"""
        cache_path = self._get_cache_path(cache_key)

        try:
            with open(cache_path, "wb") as f:
                pickle.dump(value, f, protocol=pickle.HIGHEST_PROTOCOL)
        except (pickle.PickleError, IOError, OSError):
            pass

    def __contains__(self, cache_key: str) -> bool:
        """Check if cache exists"""
        return self._cache_key_exists(cache_key)

    def __call__(
        self,
        func: Optional[Callable] = None,
        *,
        key_params: List[str],
        enabled: bool = True,
    ):
        """Make ResultCache usable as decorator

        Args:
            func: Decorated function (passed by interpreter)
            key_params: List of parameter names that participate in cache key generation (required)
            enabled: Whether to enable caching

        Returns:
            Decorator function or wrapped function
        """
        # Handle case with parentheses decorator: @cache(key_params=["x"])
        if func is None:

            def decorator(f: Callable) -> Callable:
                return self._create_wrapper(f, key_params, enabled)

            return decorator

        # Handle case without parentheses decorator: @cache
        return self._create_wrapper(func, key_params, enabled)

    def _create_wrapper(
        self,
        func: Callable,
        key_params: List[str],
        enabled: bool = True,
    ) -> Callable:
        """Create wrapper function"""
        if not enabled:
            return func

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Generate cache key
            cache_key = self._generate_cache_key(func, args, kwargs, key_params)

            # Try to get cache
            cached_result = self.get(cache_key)
            if cached_result is not None:
                return cached_result

            # Execute function and cache result
            result = func(*args, **kwargs)
            self.set(cache_key, result)

            return result

        # Attach methods to decorated function
        wrapper.get_cache_key = lambda *a, **kw: self._generate_cache_key(
            func, a, kw, key_params
        )
        wrapper.cache_manager = self

        return wrapper
